from abc import ABC, abstractmethod
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


class VectorStoreInterface(ABC):
    @abstractmethod
    def create_from_documents(self, documents: List[Document], embeddings: Any) -> None:
        """Create vector store from a list of documents.

        Args:
            documents: List of documents to index
            embeddings: Embedding model to use
        """
        pass

    @abstractmethod
    def load_existing(self) -> None:
        """Load an existing vector store."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save the current vector store."""
        pass

    @abstractmethod
    def get_retriever(
        self, search_type: str = "mmr", k: int = 5, **kwargs
    ) -> VectorStoreRetriever:
        """Get a retriever for the vector store.

        Args:
            search_type: Type of search ("mmr" or "similarity")
            k: Number of documents to retrieve
            **kwargs: Additional search parameters

        Returns:
            VectorStoreRetriever instance
        """
        pass

    @abstractmethod
    def exists(self) -> bool:
        """Check if the vector store exists.

        Returns:
            True if vector store exists, False otherwise
        """
        pass

    @abstractmethod
    def delete(self) -> None:
        """Delete the vector store."""
        pass

    @property
    @abstractmethod
    def store_type(self) -> str:
        """Get the store type identifier.

        Returns:
            String identifier for the store type
        """
        pass
