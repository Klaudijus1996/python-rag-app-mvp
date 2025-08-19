import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import DataConverter
from vector_stores import VectorStoreFactory, VectorStoreInterface

load_dotenv()

# Configuration
DATA_PATH = "data/big-basket-products-28k.csv"
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def row_to_text(row: pd.Series) -> str:
    """Convert a product row to formatted text for embedding."""
    
    # Core product information
    core_info = f"""
Name: {row.get('product', '')}
Category: {row.get('category', '')}
Sub Category: {row.get('sub_category', '')}
Brand: {row.get('brand', '')}
Price: {row.get('sale_price', '')}
Type: {row.get('type', '')}
Rating: {row.get('rating', '')}
Description: {row.get('description', '')}
"""
    
    return core_info.strip()


def create_document_metadata(row: pd.Series) -> Dict[str, Any]:
    """Extract structured metadata from product row."""
    
    metadata = {
        "product_id": DataConverter.to_string(row.get("index")),
        "name": DataConverter.to_string(row.get("product")),
        "brand": DataConverter.to_string(row.get("brand")),
        "category": DataConverter.to_string(row.get("category")),
        "sub_category": DataConverter.to_string(row.get("sub_category")),
        "price": DataConverter.to_float(row.get("sale_price")),
        "type": DataConverter.to_string(row.get("type")),
        "rating": DataConverter.to_float(row.get("rating"))
    }
    
    # Clean empty string values but keep 0.0 float values
    return {k: v for k, v in metadata.items() if v != ""}


def load_and_process_data(file_path: str) -> List[Document]:
    """Load CSV data and convert to LangChain documents."""
    
    logger.info(f"Loading data from {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path).fillna("")
        logger.info(f"Loaded {len(df)} products from CSV")
        
        documents = []
        for idx, row in df.iterrows():
            try:
                content = row_to_text(row)
                metadata = create_document_metadata(row)
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Failed to process row {idx}: {e}")
                continue
        
        logger.info(f"Successfully created {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for better retrieval."""
    
    logger.info(f"Chunking {len(documents)} documents")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    return chunks


def create_and_save_vector_store(chunks: List[Document]) -> VectorStoreInterface:
    """Create and save vector store from document chunks using abstracted interface."""
    
    logger.info("Creating embeddings and vector store using abstracted interface")
    
    try:
        # Create vector store using factory pattern
        vector_store = VectorStoreFactory.create_from_env()
        logger.info(f"Using vector store type: {vector_store.store_type}")
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        
        # Create vector store from documents
        vector_store.create_from_documents(chunks, embeddings)
        
        # Save the vector store (for FAISS, saves to disk; for Pinecone, auto-persisted)
        vector_store.save()
        
        logger.info("Vector store created and saved successfully")
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise


def main():
    """Main ingestion pipeline."""
    
    logger.info("Starting ingestion pipeline")
    
    try:
        # Load and process data
        documents = load_and_process_data(DATA_PATH)
        
        if not documents:
            logger.error("No documents to process")
            return
        
        # Chunk documents
        chunks = chunk_documents(documents)
        
        # Create and save vector store using abstracted interface
        vector_store = create_and_save_vector_store(chunks)
        
        logger.info("Ingestion completed successfully!")
        logger.info(f"Processed {len(documents)} products into {len(chunks)} chunks")
        logger.info(f"Using {vector_store.store_type} vector store")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()