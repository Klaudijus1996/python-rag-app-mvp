"""Configuration management for vector stores."""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class FAISSConfig:
    """Configuration for FAISS vector store."""
    index_dir: str


@dataclass
class PineconeConfig:
    """Configuration for Pinecone vector store."""
    api_key: str
    region: str
    cloud: str
    index_name: str
    dimension: int = 1536
    metric: str = "cosine"
    
    def validate(self) -> None:
        """Validate Pinecone configuration."""
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required when using Pinecone vector store")
        if not self.region:
            raise ValueError("PINECONE_REGION is required when using Pinecone vector store")
        if not self.cloud:
            raise ValueError("PINECONE_CLOUD is required when using Pinecone vector store")
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME is required when using Pinecone vector store")


@dataclass
class VectorStoreConfig:
    """Main configuration for vector store selection and setup."""
    store_type: str
    embedding_model: str
    faiss_config: Optional[FAISSConfig] = None
    pinecone_config: Optional[PineconeConfig] = None
    
    @classmethod
    def from_env(cls) -> 'VectorStoreConfig':
        """Create configuration from environment variables.
        
        Returns:
            VectorStoreConfig instance
            
        Raises:
            ValueError: If required configuration is missing
        """
        store_type = os.getenv("VECTOR_STORE_TYPE", "faiss").lower()
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        logger.info(f"Configuring vector store type: {store_type}")
        
        config = cls(store_type=store_type, embedding_model=embedding_model)
        
        if store_type == "faiss":
            config.faiss_config = FAISSConfig(
                index_dir=os.getenv("FAISS_INDEX_DIR", "store/faiss")
            )
            logger.info(f"FAISS config: index_dir={config.faiss_config.index_dir}")
            
        elif store_type == "pinecone":
            config.pinecone_config = PineconeConfig(
                api_key=os.getenv("PINECONE_API_KEY", ""),
                region=os.getenv("PINECONE_REGION", ""),
                cloud=os.getenv("PINECONE_CLOUD", ""),
                index_name=os.getenv("PINECONE_INDEX_NAME", "grocery-rag-index"),
                dimension=int(os.getenv("PINECONE_DIMENSION", "1536")),
                metric=os.getenv("PINECONE_METRIC", "cosine")
            )
            config.pinecone_config.validate()
            logger.info(f"Pinecone config: index_name={config.pinecone_config.index_name}, "
                       f"region={config.pinecone_config.region}, cloud={config.pinecone_config.cloud}, "
                       f"dimension={config.pinecone_config.dimension}, metric={config.pinecone_config.metric}")
            
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}. Supported types: faiss, pinecone")
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for factory pattern.
        
        Returns:
            Dictionary representation of configuration
        """
        result = {
            "store_type": self.store_type,
            "embedding_model": self.embedding_model
        }
        
        if self.faiss_config:
            result.update({
                "index_dir": self.faiss_config.index_dir
            })
        
        if self.pinecone_config:
            result.update({
                "api_key": self.pinecone_config.api_key,
                "region": self.pinecone_config.region,
                "cloud": self.pinecone_config.cloud,
                "index_name": self.pinecone_config.index_name,
                "dimension": self.pinecone_config.dimension,
                "metric": self.pinecone_config.metric
            })
        
        return result