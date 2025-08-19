"""FAISS vector store implementation."""

import logging
from pathlib import Path
from typing import List, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from .base import VectorStoreInterface

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStoreInterface):
    """FAISS vector store implementation following the VectorStoreInterface."""
    
    def __init__(self, index_dir: str, embeddings: Any):
        """Initialize FAISS vector store.
        
        Args:
            index_dir: Directory to store FAISS index files
            embeddings: Embedding model to use
        """
        self.index_dir = index_dir
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None
        
        logger.info(f"Initialized FAISS vector store with index_dir: {index_dir}")
    
    def create_from_documents(self, documents: List[Document], embeddings: Any) -> None:
        """Create FAISS vector store from documents.
        
        Args:
            documents: List of documents to index
            embeddings: Embedding model to use
        """
        logger.info(f"Creating FAISS vector store from {len(documents)} documents")
        
        try:
            self.vector_store = FAISS.from_documents(documents, embeddings)
            logger.info("FAISS vector store created successfully")
        except Exception as e:
            logger.error(f"Failed to create FAISS vector store: {e}", exc_info=True)
            raise
    
    def load_existing(self) -> None:
        """Load existing FAISS vector store from disk."""
        if not self.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_dir}")
        
        logger.info(f"Loading FAISS vector store from {self.index_dir}")
        
        try:
            self.vector_store = FAISS.load_local(
                self.index_dir, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS vector store loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FAISS vector store: {e}", exc_info=True)
            raise
    
    def save(self) -> None:
        """Save FAISS vector store to disk."""
        if not self.vector_store:
            raise RuntimeError("No vector store to save. Create or load a vector store first.")
        
        logger.info(f"Saving FAISS vector store to {self.index_dir}")
        
        try:
            # Create directory if it doesn't exist
            Path(self.index_dir).mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(self.index_dir)
            logger.info("FAISS vector store saved successfully")
        except Exception as e:
            logger.error(f"Failed to save FAISS vector store: {e}", exc_info=True)
            raise
    
    def get_retriever(self, search_type: str = "mmr", k: int = 5, **kwargs) -> VectorStoreRetriever:
        """Get retriever for FAISS vector store.
        
        Args:
            search_type: Type of search ("mmr" or "similarity")
            k: Number of documents to retrieve
            **kwargs: Additional search parameters
            
        Returns:
            VectorStoreRetriever instance
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Load or create a vector store first.")
        
        logger.info(f"Creating FAISS retriever with search_type={search_type}, k={k}")
        
        try:
            if search_type == "mmr":
                retriever = self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,
                        "fetch_k": min(20, 4 * k),
                        "lambda_mult": kwargs.get("lambda_mult", 0.4),
                        **{key: value for key, value in kwargs.items() if key != "lambda_mult"}
                    }
                )
            else:
                retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k, **kwargs}
                )
            
            logger.info("FAISS retriever created successfully")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to create FAISS retriever: {e}", exc_info=True)
            raise
    
    def exists(self) -> bool:
        """Check if FAISS index exists.
        
        Returns:
            True if index directory exists and contains files, False otherwise
        """
        index_path = Path(self.index_dir)
        if not index_path.exists():
            return False
        
        # Check if directory contains any files (FAISS creates .faiss and .pkl files)
        return any(index_path.iterdir())
    
    def delete(self) -> None:
        """Delete FAISS vector store files."""
        index_path = Path(self.index_dir)
        
        if index_path.exists():
            logger.info(f"Deleting FAISS vector store at {self.index_dir}")
            
            try:
                # Remove all files in the index directory
                for file_path in index_path.glob("*"):
                    file_path.unlink()
                
                # Remove directory if empty
                if not any(index_path.iterdir()):
                    index_path.rmdir()
                
                logger.info("FAISS vector store deleted successfully")
            except Exception as e:
                logger.error(f"Failed to delete FAISS vector store: {e}", exc_info=True)
                raise
        else:
            logger.warning(f"FAISS index directory {self.index_dir} does not exist")
    
    @property
    def store_type(self) -> str:
        """Get store type identifier."""
        return "faiss"