import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from dotenv import load_dotenv

from schema import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    QueryType,
)
from chains import RAGSystem
import ingest

# Import our logging system
from logging_config import setup_logging, get_logger, cleanup_logging
from middleware import (
    RequestLoggingMiddleware,
    PerformanceMonitoringMiddleware,
    ErrorTrackingMiddleware,
)

# Load environment variables
load_dotenv()

# Configuration
APP_ENV = os.getenv("APP_ENV", "dev")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
INDEX_DIR = "store/faiss"

# Setup logging
setup_logging(
    app_env=APP_ENV,
    use_async=True,
    use_json=APP_ENV == "production",  # Use JSON logs in production
)
logger = get_logger(__name__)

# Global RAG system instance
rag_system: Optional[RAGSystem] = None

# Global ingestion status tracking
ingestion_status: Dict[str, Any] = {
    "status": "idle",  # idle, running, completed, failed
    "progress": 0,
    "message": "No ingestion in progress",
    "start_time": None,
    "end_time": None,
    "chunks_indexed": 0,
    "products_processed": 0,
    "error": None
}

# Semaphore to limit concurrent expensive operations
max_concurrent_queries = int(os.getenv("MAX_CONCURRENT_QUERIES", "3"))
query_semaphore = asyncio.Semaphore(max_concurrent_queries)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global rag_system

    # Startup
    logger.info("Starting up RAG application")
    try:
        if check_index_exists():
            rag_system = RAGSystem()
            await rag_system.initialize()
            logger.info("RAG system initialized successfully")
        else:
            logger.warning("Vector index not found. Run ingestion first.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
        # Continue startup without RAG system

    yield

    # Shutdown
    logger.info("Shutting down RAG application")
    cleanup_logging()  # Clean up async logging handlers


# FastAPI app
app = FastAPI(
    title="Retail RAG MVP",
    description="A production-ready RAG application for product recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# Add logging middleware
app.add_middleware(
    RequestLoggingMiddleware,
    log_request_body=APP_ENV == "dev",  # Only log request bodies in development
    log_response_body=False,  # Disable response body logging for performance
    exclude_paths=["/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"],
)

app.add_middleware(
    PerformanceMonitoringMiddleware,
    slow_request_threshold=5.0,  # Log requests taking longer than 5 seconds
)

app.add_middleware(ErrorTrackingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_rag_system() -> RAGSystem:
    """Dependency to get RAG system instance."""
    global rag_system

    # If RAG system doesn't exist, try to initialize it if index is available
    if rag_system is None and check_index_exists():
        try:
            rag_system = RAGSystem()
            await rag_system.initialize()
            logger.info("RAG system initialized on-demand after external ingestion")
        except Exception as e:
            logger.error(
                f"Failed to initialize RAG system on-demand: {e}", exc_info=True
            )

    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not available. Please run ingestion first.",
        )
    return rag_system


def get_index_path_for_response(vector_store=None) -> str:
    """Get the appropriate index path for API responses based on vector store type.

    Args:
        vector_store: Optional vector store instance to get the actual type from.
                     If None, reads from environment variables.
    """
    if vector_store:
        store_type = vector_store.store_type
    else:
        store_type = os.getenv("VECTOR_STORE_TYPE", "faiss").lower()

    if store_type == "faiss":
        return os.getenv("FAISS_INDEX_DIR", "store/faiss")
    elif store_type == "pinecone":
        return f"pinecone://{os.getenv('PINECONE_INDEX_NAME', 'grocery-rag-index')}"
    else:
        return "unknown"


def check_index_exists() -> bool:
    """Check if vector index exists using vector store abstraction."""
    try:
        from vector_stores import VectorStoreFactory

        vector_store = VectorStoreFactory.create_from_env()
        return vector_store.exists()
    except Exception:
        # Fallback to checking the hardcoded FAISS path for backward compatibility
        index_path = Path(INDEX_DIR)
        return index_path.exists() and any(index_path.iterdir())


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect."""
    return {"message": "Retail RAG MVP is running. Visit /docs for API documentation."}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    index_ready = check_index_exists()
    total_chunks = None

    if index_ready and rag_system:
        try:
            # Try to get some basic info about the index
            # For now, we'll use None since we don't have an easy way to count chunks
            # This could be implemented by accessing the vector store metadata
            total_chunks = None  # Could implement actual count later
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if index_ready else "degraded",
        version="1.0.0",
        index_ready=index_ready,
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        total_chunks=total_chunks,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, rag: RAGSystem = Depends(get_rag_system)):
    """Main chat endpoint for product queries."""
    # Use semaphore to limit concurrent processing
    async with query_semaphore:
        try:
            start_time = time.time()

            # Validate request
            if not request.query.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty"
                )

            # Process query
            logger.info(
                f"Processing query for session {request.session_id}: {request.query[:100]}..."
            )

            # Get response from RAG system (now async)
            answer = await rag.query(request.query, request.session_id)

            # Get relevant products for additional context (now async)
            retrieval_result = await rag.retrieve(request.query)
            products = retrieval_result.get("products", [])
            detected_query_type = retrieval_result.get("query_type", QueryType.INFORMATION)

            # Apply filters if provided
            if request.category_filter:
                products = [
                    p
                    for p in products
                    if request.category_filter.lower() in p.category.lower()
                ]

            if request.price_range:
                min_price = request.price_range.get("min", 0)
                max_price = request.price_range.get("max", float("inf"))
                products = [p for p in products if min_price <= p.price <= max_price]

            # Limit number of products
            products = products[: request.max_products]

            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f}s")

            return ChatResponse(
                answer=answer,
                products=products[: request.max_products] if products else None,
                query_type=request.query_type or detected_query_type,
                sources_count=len(retrieval_result.get("documents", [])),
                session_id=request.session_id,
            )

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
            )
        except Exception as e:
            logger.error(f"Chat processing error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while processing your request",
            )


async def run_ingestion_background(request: IngestRequest):
    """Background task for data ingestion."""
    global ingestion_status, rag_system
    
    try:
        # Update status
        ingestion_status.update({
            "status": "running",
            "progress": 0,
            "message": "Starting ingestion process",
            "start_time": datetime.now(),
            "end_time": None,
            "error": None
        })
        
        logger.info("Starting background data ingestion process")
        
        # Set environment variables for ingestion
        os.environ["RAG_CHUNK_SIZE"] = str(request.chunk_size)
        os.environ["RAG_CHUNK_OVERLAP"] = str(request.chunk_overlap)
        
        # Update progress
        ingestion_status.update({"progress": 20, "message": "Loading and processing data"})
        
        # Run ingestion operations in thread pool to avoid blocking
        documents = await asyncio.to_thread(ingest.load_and_process_data, ingest.DATA_PATH)
        
        ingestion_status.update({"progress": 50, "message": "Chunking documents"})
        chunks = await asyncio.to_thread(ingest.chunk_documents, documents)
        
        ingestion_status.update({"progress": 80, "message": "Creating vector store"})
        vector_store = await asyncio.to_thread(ingest.create_and_save_vector_store, chunks)
        
        # Reinitialize RAG system with new index
        try:
            if rag_system:
                rag_system = RAGSystem()
                await rag_system.initialize()
                logger.info("RAG system reinitialized with new index")
        except Exception as e:
            logger.warning(f"Failed to reinitialize RAG system: {e}")
        
        # Update final status
        ingestion_status.update({
            "status": "completed",
            "progress": 100,
            "message": "Ingestion completed successfully",
            "end_time": datetime.now(),
            "chunks_indexed": len(chunks),
            "products_processed": len(documents)
        })
        
        logger.info(f"Background ingestion completed: {len(chunks)} chunks, {len(documents)} products")
        
    except Exception as e:
        logger.error(f"Background ingestion error: {e}", exc_info=True)
        ingestion_status.update({
            "status": "failed",
            "message": f"Ingestion failed: {str(e)}",
            "end_time": datetime.now(),
            "error": str(e)
        })

@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(background_tasks: BackgroundTasks, request: IngestRequest = IngestRequest()):
    """Trigger data ingestion and vector store creation as background task."""
    try:
        logger.info("Received ingestion request")

        # Check if ingestion is already running
        if ingestion_status["status"] == "running":
            return IngestResponse(
                status="already_running",
                chunks_indexed=ingestion_status.get("chunks_indexed", 0),
                products_processed=ingestion_status.get("products_processed", 0),
                index_path=get_index_path_for_response(),
                embedding_model=os.getenv(
                    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
                ),
                processing_time_seconds=0,
            )

        # Check if index exists and force_reindex is False
        if check_index_exists() and not request.force_reindex:
            return IngestResponse(
                status="already_exists",
                chunks_indexed=0,
                index_path=get_index_path_for_response(),
                embedding_model=os.getenv(
                    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
                ),
                processing_time_seconds=0,
            )

        # Start background ingestion
        background_tasks.add_task(run_ingestion_background, request)
        
        return IngestResponse(
            status="started",
            chunks_indexed=0,
            products_processed=0,
            index_path=get_index_path_for_response(),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            processing_time_seconds=0,
        )

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Product data file not found"
        )
    except Exception as e:
        logger.error(f"Ingestion request error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start ingestion: {str(e)}",
        )


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str, rag: RAGSystem = Depends(get_rag_system)):
    """Get information about a user session."""
    try:
        session_info = rag.get_session_info(session_id)
        return session_info
    except Exception as e:
        logger.error(f"Error getting session info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session information",
        )


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str, rag: RAGSystem = Depends(get_rag_system)):
    """Clear a user session."""
    try:
        success = rag.clear_session(session_id)
        if success:
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            return {"message": f"Session {session_id} not found"}
    except Exception as e:
        logger.error(f"Error clearing session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear session",
        )


@app.get("/retrieve/{query}")
async def retrieve_documents(query: str, rag: RAGSystem = Depends(get_rag_system)):
    """Retrieve relevant documents without generating a response."""
    async with query_semaphore:
        try:
            result = await rag.retrieve(query)
            return {
                "query": query,
                "query_type": result["query_type"],
                "products": result["products"],
                "document_count": len(result["documents"]),
            }
        except Exception as e:
            logger.error(f"Document retrieval error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve documents",
            )


@app.get("/ingest/status")
async def get_ingestion_status():
    """Get current ingestion status and progress."""
    return ingestion_status


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "details": {"status_code": exc.status_code},
        },
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "error_code": "VALIDATION_ERROR",
            "details": {"errors": exc.errors()},
        },
    )


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {APP_HOST}:{APP_PORT}")
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "dev",
        log_level="info",
    )
