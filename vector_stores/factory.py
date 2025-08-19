"""Factory pattern for creating vector store instances."""

import logging

from langchain_openai import OpenAIEmbeddings

from .base import VectorStoreInterface
from .faiss_store import FAISSVectorStore
from .pinecone_store import PineconeVectorStoreWrapper
from .config import VectorStoreConfig

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory for creating vector store instances following the Factory pattern."""
    
    @staticmethod
    def create_vector_store(config: VectorStoreConfig) -> VectorStoreInterface:
        """Create a vector store instance based on configuration.
        
        Args:
            config: VectorStoreConfig instance
            
        Returns:
            VectorStoreInterface implementation
            
        Raises:
            ValueError: If unsupported store type is specified
            ImportError: If required dependencies are not available
        """
        store_type = config.store_type.lower()
        
        logger.info(f"Creating vector store of type: {store_type}")
        
        # Create embeddings model
        embeddings = OpenAIEmbeddings(model=config.embedding_model)
        logger.info(f"Using embedding model: {config.embedding_model}")
        
        if store_type == "faiss":
            if not config.faiss_config:
                raise ValueError("FAISS configuration is required for FAISS vector store")
            
            return FAISSVectorStore(
                index_dir=config.faiss_config.index_dir,
                embeddings=embeddings
            )
        
        elif store_type == "pinecone":
            if not config.pinecone_config:
                raise ValueError("Pinecone configuration is required for Pinecone vector store")
            
            # Convert config to dictionary for Pinecone store
            pinecone_config_dict = {
                "api_key": config.pinecone_config.api_key,
                "region": config.pinecone_config.region,
                "cloud": config.pinecone_config.cloud,
                "dimension": config.pinecone_config.dimension,
                "metric": config.pinecone_config.metric
            }
            
            return PineconeVectorStoreWrapper(
                index_name=config.pinecone_config.index_name,
                embeddings=embeddings,
                config=pinecone_config_dict
            )
        
        else:
            supported_types = ["faiss", "pinecone"]
            raise ValueError(
                f"Unsupported vector store type: {store_type}. "
                f"Supported types: {', '.join(supported_types)}"
            )
    
    @staticmethod
    def create_from_env() -> VectorStoreInterface:
        """Create vector store from environment variables.
        
        Returns:
            VectorStoreInterface implementation
        """
        config = VectorStoreConfig.from_env()
        return VectorStoreFactory.create_vector_store(config)
    
    @staticmethod
    def get_supported_types() -> list[str]:
        """Get list of supported vector store types.
        
        Returns:
            List of supported store type strings
        """
        return ["faiss", "pinecone"]