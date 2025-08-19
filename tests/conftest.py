"""Shared pytest configuration and fixtures for the RAG MVP test suite."""

import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from typing import Generator, List

import pytest
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from vector_stores.base import VectorStoreInterface
from vector_stores.config import VectorStoreConfig, FAISSConfig, PineconeConfig


@pytest.fixture
def setup_test_environment():
    """Set up clean test environment variables when explicitly requested."""
    # Store original values
    original_env = {}
    
    # Environment variables to set/override for tests
    test_env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
        "RAG_CHUNK_SIZE": "1000",
        "RAG_CHUNK_OVERLAP": "200",
        "RAG_TOP_K_RESULTS": "5",
        "VECTOR_STORE_TYPE": "faiss",
        "FAISS_INDEX_DIR": "test_store/faiss",
        "PINECONE_API_KEY": "test-pinecone-key",
        "PINECONE_INDEX_NAME": "test-index",
        "PINECONE_REGION": "us-east-1",
        "PINECONE_CLOUD": "aws",
        "PINECONE_DIMENSION": "1536",
        "PINECONE_METRIC": "cosine"
    }
    
    # Store original values and set test values
    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings."""
    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings_class:
        mock_embeddings = Mock(spec=OpenAIEmbeddings)
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3] * 512]  # 1536-dim vector
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3] * 512
        mock_embeddings_class.return_value = mock_embeddings
        yield mock_embeddings


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock(spec=VectorStoreInterface)
    mock_store.store_type = "faiss"
    mock_store.exists.return_value = True
    mock_store.create_from_documents.return_value = None
    mock_store.load_existing.return_value = None
    mock_store.save.return_value = None
    mock_store.delete.return_value = None
    
    # Mock retriever
    mock_retriever = Mock()
    mock_retriever.get_relevant_documents.return_value = []
    mock_retriever.invoke.return_value = []
    mock_store.get_retriever.return_value = mock_retriever
    
    return mock_store


@pytest.fixture
def sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="High-quality organic apples from local farms, perfect for healthy snacking",
            metadata={
                "product_id": "1",
                "name": "Organic Red Apples",
                "brand": "Fresh Farm",
                "category": "Fruits & Vegetables",
                "sub_category": "Fresh Fruits",
                "price": 2.99,
                "type": "Organic",
                "rating": 4.5,
                "description": "Premium organic red apples"
            }
        ),
        Document(
            page_content="Premium storage containers with airtight seals for kitchen organization",
            metadata={
                "product_id": "2",
                "name": "Storage Container Set",
                "brand": "Nakoda",
                "category": "Cleaning & Household",
                "sub_category": "Storage & Accessories",
                "price": 149.0,
                "type": "Storage Baskets",
                "rating": 4.2,
                "description": "Professional storage container set"
            }
        ),
        Document(
            page_content="Effective multipurpose cleaning wipes for all surfaces",
            metadata={
                "product_id": "3",
                "name": "Multipurpose Cleaning Wipes",
                "brand": "Nature Protect",
                "category": "Cleaning & Household",
                "sub_category": "All Purpose Cleaners",
                "price": 169.0,
                "type": "Disinfectant Spray & Cleaners",
                "rating": 3.8,
                "description": "Eco-friendly cleaning wipes"
            }
        )
    ]


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for testing ingestion."""
    return pd.DataFrame({
        'index': [1, 2, 3],
        'product': ['Organic Red Apples', 'Storage Container Set', 'Multipurpose Cleaning Wipes'],
        'brand': ['Fresh Farm', 'Nakoda', 'Nature Protect'],
        'category': ['Fruits & Vegetables', 'Cleaning & Household', 'Cleaning & Household'],
        'sub_category': ['Fresh Fruits', 'Storage & Accessories', 'All Purpose Cleaners'],
        'sale_price': [2.99, 149.0, 169.0],
        'type': ['Organic', 'Storage Baskets', 'Disinfectant Spray & Cleaners'],
        'rating': [4.5, 4.2, 3.8],
        'description': [
            'Premium organic red apples',
            'Professional storage container set',
            'Eco-friendly cleaning wipes'
        ]
    })


@pytest.fixture
def faiss_config():
    """Create FAISS configuration for testing."""
    return VectorStoreConfig(
        store_type="faiss",
        embedding_model="text-embedding-3-small",
        faiss_config=FAISSConfig(index_dir="test_store/faiss")
    )


@pytest.fixture
def pinecone_config():
    """Create Pinecone configuration for testing."""
    return VectorStoreConfig(
        store_type="pinecone",
        embedding_model="text-embedding-3-small",
        pinecone_config=PineconeConfig(
            api_key="test-api-key",
            region="us-east-1",
            cloud="aws",
            index_name="test-index",
            dimension=1536,
            metric="cosine"
        )
    )


@pytest.fixture
def mock_faiss():
    """Mock FAISS vector store components."""
    with patch('vector_stores.faiss_store.FAISS') as mock_faiss_class:
        mock_faiss_instance = Mock()
        mock_faiss_instance.save_local = Mock()
        mock_faiss_instance.as_retriever = Mock()
        
        mock_faiss_class.from_documents.return_value = mock_faiss_instance
        mock_faiss_class.load_local.return_value = mock_faiss_instance
        
        yield mock_faiss_class, mock_faiss_instance


@pytest.fixture
def mock_pinecone():
    """Mock Pinecone vector store components."""
    with patch('vector_stores.pinecone_store.PINECONE_AVAILABLE', True), \
         patch('vector_stores.pinecone_store.Pinecone') as mock_pinecone_class, \
         patch('vector_stores.pinecone_store.PineconeVectorStore') as mock_pinecone_vector_store:
        
        # Mock Pinecone client
        mock_pinecone_instance = Mock()
        mock_pinecone_instance.list_indexes.return_value.indexes = []
        mock_pinecone_instance.create_index = Mock()
        mock_pinecone_instance.delete_index = Mock()
        mock_pinecone_class.return_value = mock_pinecone_instance
        
        # Mock PineconeVectorStore
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.as_retriever = Mock()
        mock_pinecone_vector_store.from_documents.return_value = mock_vector_store_instance
        mock_pinecone_vector_store.from_existing_index.return_value = mock_vector_store_instance
        
        yield mock_pinecone_class, mock_pinecone_instance, mock_pinecone_vector_store, mock_vector_store_instance


@pytest.fixture
def mock_chat_openai():
    """Mock ChatOpenAI for testing."""
    with patch('chains.ChatOpenAI') as mock_chat_class:
        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = "This is a test response from ChatOpenAI"
        mock_chat_class.return_value = mock_chat_instance
        yield mock_chat_instance


@pytest.fixture
def clean_environment():
    """Ensure clean environment state for each test."""
    # Clear any cached modules
    import sys
    modules_to_clear = [name for name in sys.modules.keys() if name.startswith(('vector_stores', 'chains', 'ingest'))]
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    yield
    
    # Clean up after test
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]


@pytest.fixture
def test_data_file(temp_dir, sample_dataframe) -> str:
    """Create a test CSV file with sample data."""
    file_path = os.path.join(temp_dir, "test_data.csv")
    sample_dataframe.to_csv(file_path, index=False)
    return file_path


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may be slow)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast)"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their names and locations."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "Integration" in str(item.cls):
            item.add_marker(pytest.mark.integration)
        # Mark unit tests
        elif "unit" in item.nodeid.lower() or "Unit" in str(item.cls):
            item.add_marker(pytest.mark.unit)