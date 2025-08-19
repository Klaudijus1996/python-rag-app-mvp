"""Pinecone vector store implementation using langchain-pinecone."""

import logging
import time
from typing import List, Any, Optional

from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from .base import VectorStoreInterface

logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone dependencies not available. Install langchain-pinecone to use Pinecone vector store.")


class PineconeVectorStoreWrapper(VectorStoreInterface):
    """Pinecone vector store implementation using langchain-pinecone."""
    
    def __init__(self, index_name: str, embeddings: Any, config: dict):
        """Initialize Pinecone vector store.
        
        Args:
            index_name: Name of the Pinecone index
            embeddings: Embedding model to use  
            config: Configuration dictionary containing Pinecone settings
        """
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone dependencies not available. Install langchain-pinecone to use Pinecone.")
        
        self.index_name = index_name
        self.embeddings = embeddings
        self.api_key = config.get("api_key")
        self.region = config.get("region")
        self.cloud = config.get("cloud")
        self.dimension = config.get("dimension", 1536)
        self.metric = config.get("metric", "cosine")
        self.vector_store: Optional[PineconeVectorStore] = None
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        logger.info(f"Initialized Pinecone vector store: index_name={index_name}, "
                   f"dimension={self.dimension}, metric={self.metric}")
    
    def create_from_documents(self, documents: List[Document], embeddings: Any) -> None:
        """Create Pinecone vector store from documents.
        
        Args:
            documents: List of documents to index
            embeddings: Embedding model to use
        """
        logger.info(f"Creating Pinecone vector store from {len(documents)} documents")
        
        try:
            # Create index if it doesn't exist
            if not self.exists():
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
                # Wait for index to be ready
                self._wait_for_index_ready()
            
            # Create vector store from documents
            self.vector_store = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=self.index_name,
                pinecone_api_key=self.api_key
            )
            
            logger.info("Pinecone vector store created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone vector store: {e}", exc_info=True)
            raise
    
    def load_existing(self) -> None:
        """Load existing Pinecone vector store."""
        if not self.exists():
            raise ValueError(f"Pinecone index '{self.index_name}' does not exist")
        
        logger.info(f"Loading Pinecone vector store: {self.index_name}")
        
        try:
            self.vector_store = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
            )
            logger.info("Pinecone vector store loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Pinecone vector store: {e}", exc_info=True)
            raise
    
    def save(self) -> None:
        """Save operation is not needed for Pinecone as it auto-saves."""
        logger.info("Pinecone vector store automatically persists data - no save operation needed")
    
    def get_retriever(self, search_type: str = "mmr", k: int = 5, **kwargs) -> VectorStoreRetriever:
        """Get retriever for Pinecone vector store.
        
        Args:
            search_type: Type of search ("mmr" or "similarity")
            k: Number of documents to retrieve
            **kwargs: Additional search parameters
            
        Returns:
            VectorStoreRetriever instance
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Load or create a vector store first.")
        
        logger.info(f"Creating Pinecone retriever with search_type={search_type}, k={k}")
        
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
            
            logger.info("Pinecone retriever created successfully")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone retriever: {e}", exc_info=True)
            raise
    
    def exists(self) -> bool:
        """Check if Pinecone index exists.
        
        Returns:
            True if index exists, False otherwise
        """
        try:
            indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in indexes.indexes]
            return self.index_name in index_names
        except Exception as e:
            logger.error(f"Failed to check if Pinecone index exists: {e}")
            return False
    
    def delete(self) -> None:
        """Delete Pinecone index."""
        if not self.exists():
            logger.warning(f"Pinecone index '{self.index_name}' does not exist")
            return
        
        logger.info(f"Deleting Pinecone index: {self.index_name}")
        
        try:
            self.pc.delete_index(self.index_name)
            logger.info("Pinecone index deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index: {e}", exc_info=True)
            raise
    
    @property
    def store_type(self) -> str:
        """Get store type identifier."""
        return "pinecone"
    
    
    def _wait_for_index_ready(self, timeout: int = 60) -> None:
        """Wait for Pinecone index to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        logger.info(f"Waiting for Pinecone index '{self.index_name}' to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                index_stats = self.pc.describe_index(self.index_name)
                if index_stats.status.ready:
                    logger.info(f"Pinecone index '{self.index_name}' is ready")
                    return
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Error checking index status: {e}")
                time.sleep(2)
        
        raise TimeoutError(f"Pinecone index '{self.index_name}' did not become ready within {timeout} seconds")