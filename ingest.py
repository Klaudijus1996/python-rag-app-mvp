import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Configuration
DATA_PATH = "data/products.csv"
INDEX_DIR = "store/faiss"
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
Name: {row.get('name', '')}
Category: {row.get('category', '')}
Brand: {row.get('brand', '')}
Price: {row.get('price', '')} {row.get('currency', '')}
Description: {row.get('description', '')}
"""
    
    # Price influencers and features
    influencers = []
    for field, label in [
        ('materials', 'Materials'),
        ('capacity', 'Capacity'),
        ('features', 'Features'),
        ('compatibility', 'Compatibility'),
        ('variants', 'Variants'),
        ('tags', 'Tags')
    ]:
        value = row.get(field, '')
        if value and str(value).strip() and str(value) != 'nan':
            influencers.append(f"{label}: {value}")
    
    if influencers:
        core_info += "\n" + "\n".join(influencers)
    
    return core_info.strip()


def create_document_metadata(row: pd.Series) -> Dict[str, Any]:
    """Extract structured metadata from product row."""
    
    metadata = {
        "product_id": str(row.get("product_id", "")),
        "name": str(row.get("name", "")),
        "brand": str(row.get("brand", "")),
        "category": str(row.get("category", "")),
        "price": float(row.get("price", 0)) if pd.notna(row.get("price")) else 0.0,
        "currency": str(row.get("currency", "")),
        "sku": str(row.get("sku", "")),
        "url": str(row.get("url", "")),
        "image_url": str(row.get("image_url", "")),
        "tags": str(row.get("tags", "")),
        "materials": str(row.get("materials", "")),
        "capacity": str(row.get("capacity", "")),
        "features": str(row.get("features", "")),
        "stock": int(row.get("stock", 0)) if pd.notna(row.get("stock")) else 0
    }
    
    # Clean empty string values
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


def create_vector_store(chunks: List[Document]) -> FAISS:
    """Create FAISS vector store from document chunks."""
    
    logger.info("Creating embeddings and vector store")
    
    try:
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("Vector store created successfully")
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise


def save_vector_store(vector_store: FAISS, index_dir: str) -> None:
    """Save vector store to disk."""
    
    logger.info(f"Saving vector store to {index_dir}")
    
    try:
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        vector_store.save_local(index_dir)
        logger.info("Vector store saved successfully")
        
    except Exception as e:
        logger.error(f"Failed to save vector store: {e}")
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
        
        # Create vector store
        vector_store = create_vector_store(chunks)
        
        # Save to disk
        save_vector_store(vector_store, INDEX_DIR)
        
        logger.info(f"Ingestion completed successfully!")
        logger.info(f"Processed {len(documents)} products into {len(chunks)} chunks")
        logger.info(f"Vector store saved to: {INDEX_DIR}")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()