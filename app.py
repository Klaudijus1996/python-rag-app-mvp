import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from dotenv import load_dotenv

from schema import (
    ChatRequest, ChatResponse, IngestRequest, IngestResponse, 
    HealthResponse, ErrorResponse, QueryType
)
from chains import RAGSystem
import ingest

# Load environment variables
load_dotenv()

# Configuration
APP_ENV = os.getenv("APP_ENV", "dev")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
INDEX_DIR = "store/faiss"

# Setup logging
log_level = logging.DEBUG if APP_ENV == "dev" else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system: Optional[RAGSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global rag_system
    
    # Startup
    logger.info("Starting up RAG application")
    try:
        if check_index_exists():
            rag_system = RAGSystem()
            logger.info("RAG system initialized successfully")
        else:
            logger.warning("Vector index not found. Run ingestion first.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        # Continue startup without RAG system
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG application")


# FastAPI app
app = FastAPI(
    title="Retail RAG MVP",
    description="A production-ready RAG application for product recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_system() -> RAGSystem:
    """Dependency to get RAG system instance."""
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not available. Please run ingestion first."
        )
    return rag_system


def check_index_exists() -> bool:
    """Check if vector index exists."""
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
            total_chunks = "available"  # Could implement actual count
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if index_ready else "degraded",
        version="1.0.0",
        index_ready=index_ready,
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        total_chunks=total_chunks
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Main chat endpoint for product queries."""
    try:
        start_time = time.time()
        
        # Validate request
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Process query
        logger.info(f"Processing query for session {request.session_id}: {request.query[:100]}...")
        
        # Get response from RAG system
        answer = rag.query(request.query, request.session_id)
        
        # Get relevant products for additional context
        retrieval_result = rag.retrieve(request.query)
        products = retrieval_result.get("products", [])
        detected_query_type = retrieval_result.get("query_type", QueryType.INFORMATION)
        
        # Apply filters if provided
        if request.category_filter:
            products = [p for p in products if request.category_filter.lower() in p.category.lower()]
        
        if request.price_range:
            min_price = request.price_range.get("min", 0)
            max_price = request.price_range.get("max", float("inf"))
            products = [p for p in products if min_price <= p.price <= max_price]
        
        # Limit number of products
        products = products[:request.max_products]
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        return ChatResponse(
            answer=answer,
            products=products[:request.max_products] if products else None,
            query_type=request.query_type or detected_query_type,
            sources_count=len(retrieval_result.get("documents", [])),
            session_id=request.session_id
        )
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(request: IngestRequest = IngestRequest()):
    """Trigger data ingestion and vector store creation."""
    try:
        start_time = time.time()
        
        logger.info("Starting data ingestion process")
        
        # Check if index exists and force_reindex is False
        if check_index_exists() and not request.force_reindex:
            return IngestResponse(
                status="already_exists",
                chunks_indexed=0,
                index_path=INDEX_DIR,
                embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                processing_time_seconds=0
            )
        
        # Set environment variables for ingestion
        os.environ["RAG_CHUNK_SIZE"] = str(request.chunk_size)
        os.environ["RAG_CHUNK_OVERLAP"] = str(request.chunk_overlap)
        
        # Run ingestion
        documents = ingest.load_and_process_data(ingest.DATA_PATH)
        chunks = ingest.chunk_documents(documents)
        vector_store = ingest.create_vector_store(chunks)
        ingest.save_vector_store(vector_store, INDEX_DIR)
        
        processing_time = time.time() - start_time
        
        # Re-initialize RAG system if it exists
        global rag_system
        if rag_system:
            try:
                rag_system = RAGSystem()
                logger.info("RAG system re-initialized after ingestion")
            except Exception as e:
                logger.error(f"Failed to re-initialize RAG system: {e}")
        
        logger.info(f"Ingestion completed in {processing_time:.2f}s")
        
        return IngestResponse(
            status="completed",
            chunks_indexed=len(chunks),
            products_processed=len(documents),
            index_path=INDEX_DIR,
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            processing_time_seconds=processing_time
        )
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product data file not found"
        )
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@app.get("/sessions/{session_id}")
async def get_session_info(
    session_id: str,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Get information about a user session."""
    try:
        session_info = rag.get_session_info(session_id)
        return session_info
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session information"
        )


@app.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Clear a user session."""
    try:
        success = rag.clear_session(session_id)
        if success:
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            return {"message": f"Session {session_id} not found"}
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear session"
        )


@app.get("/retrieve/{query}")
async def retrieve_documents(
    query: str,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Retrieve relevant documents without generating a response."""
    try:
        result = rag.retrieve(query)
        return {
            "query": query,
            "query_type": result["query_type"],
            "products": result["products"],
            "document_count": len(result["documents"])
        }
    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "details": {"status_code": exc.status_code}
        }
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "error_code": "VALIDATION_ERROR",
            "details": {"errors": exc.errors()}
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {APP_HOST}:{APP_PORT}")
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "dev",
        log_level="info"
    )