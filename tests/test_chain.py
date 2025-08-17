import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd

from chains import RAGSystem, detect_query_type, format_docs, extract_products_from_docs
from schema import QueryType, ProductInfo
from langchain_core.documents import Document


class TestQueryTypeDetection:
    """Test query type detection functionality."""
    
    def test_recommendation_detection(self):
        queries = [
            "I need a good laptop",
            "Can you recommend a water bottle?",
            "What's the best smartphone under 500?",
            "Help me find a backpack",
            "Looking for workout equipment"
        ]
        for query in queries:
            assert detect_query_type(query) == QueryType.RECOMMENDATION
    
    def test_comparison_detection(self):
        queries = [
            "Compare iPhone vs Samsung",
            "What's the difference between these laptops?",
            "Which is better: laptop A or laptop B?",
            "MacBook vs ThinkPad comparison",
            "Show me the differences between these products"
        ]
        for query in queries:
            assert detect_query_type(query) == QueryType.COMPARISON
    
    def test_complement_detection(self):
        queries = [
            "What accessories go with this laptop?",
            "What pairs well with this coffee grinder?",
            "Compatible products for iPhone",
            "What works with this camera?",
            "Bundle recommendations for this item"
        ]
        for query in queries:
            assert detect_query_type(query) == QueryType.COMPLEMENT
    
    def test_information_detection(self):
        queries = [
            "Tell me about this product",
            "What are the specifications?",
            "How much does it cost?",
            "When was this released?",
            "What material is this made of?"
        ]
        for query in queries:
            assert detect_query_type(query) == QueryType.INFORMATION


class TestDocumentFormatting:
    """Test document formatting functionality."""
    
    def test_format_docs_empty(self):
        result = format_docs([])
        assert "No relevant products found" in result
    
    def test_format_docs_with_products(self):
        docs = [
            Document(
                page_content="Great laptop for work",
                metadata={
                    "name": "MacBook Pro",
                    "brand": "Apple",
                    "price": 1299.99,
                    "currency": "USD",
                    "category": "Laptops",
                    "product_id": "P001",
                    "url": "https://example.com/macbook"
                }
            ),
            Document(
                page_content="Lightweight and durable",
                metadata={
                    "name": "ThinkPad X1",
                    "brand": "Lenovo",
                    "price": 999.99,
                    "currency": "USD",
                    "category": "Laptops",
                    "product_id": "P002"
                }
            )
        ]
        
        result = format_docs(docs)
        assert "MacBook Pro" in result
        assert "Apple" in result
        assert "1299.99" in result
        assert "ThinkPad X1" in result
        assert "Lenovo" in result
        assert "https://example.com/macbook" in result


class TestProductExtraction:
    """Test product information extraction."""
    
    def test_extract_products_from_docs(self):
        docs = [
            Document(
                page_content="High-quality laptop with excellent performance",
                metadata={
                    "product_id": "P001",
                    "name": "MacBook Pro",
                    "brand": "Apple",
                    "category": "Laptops",
                    "price": 1299.99,
                    "currency": "USD",
                    "url": "https://example.com/macbook",
                    "image_url": "https://example.com/images/macbook.jpg"
                }
            )
        ]
        
        products = extract_products_from_docs(docs)
        assert len(products) == 1
        
        product = products[0]
        assert isinstance(product, ProductInfo)
        assert product.product_id == "P001"
        assert product.name == "MacBook Pro"
        assert product.brand == "Apple"
        assert product.category == "Laptops"
        assert product.price == 1299.99
        assert product.currency == "USD"
        assert product.url == "https://example.com/macbook"
    
    def test_extract_products_with_invalid_data(self):
        docs = [
            Document(
                page_content="Test content",
                metadata={
                    "product_id": "P001",
                    "name": "Test Product",
                    "brand": "Test Brand",
                    "category": "Test Category",
                    "price": "invalid_price",  # Invalid price
                    "currency": "USD"
                }
            )
        ]
        
        products = extract_products_from_docs(docs)
        # Should handle invalid data gracefully
        assert len(products) == 0


@pytest.fixture
def mock_vector_store():
    """Mock FAISS vector store for testing."""
    with patch('chains.FAISS.load_local') as mock_load:
        mock_store = Mock()
        mock_retriever = Mock()
        
        # Mock documents to return
        mock_docs = [
            Document(
                page_content="Excellent water bottle for outdoor activities",
                metadata={
                    "product_id": "P001",
                    "name": "HydroFlask 32oz",
                    "brand": "HydroFlask",
                    "category": "Drinkware",
                    "price": 39.99,
                    "currency": "USD"
                }
            )
        ]
        
        mock_retriever.get_relevant_documents.return_value = mock_docs
        mock_retriever.invoke.return_value = mock_docs
        mock_store.as_retriever.return_value = mock_retriever
        mock_load.return_value = mock_store
        
        yield mock_store


@pytest.fixture
def mock_openai():
    """Mock OpenAI components."""
    with patch('chains.OpenAIEmbeddings') as mock_embeddings, \
         patch('chains.ChatOpenAI') as mock_chat:
        
        mock_embeddings.return_value = Mock()
        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = "This is a test response"
        mock_chat.return_value = mock_chat_instance
        
        yield mock_embeddings, mock_chat


class TestRAGSystem:
    """Test the complete RAG system."""
    
    def test_rag_system_initialization(self, mock_vector_store, mock_openai):
        """Test RAG system can be initialized."""
        with patch('os.path.exists', return_value=True):
            rag_system = RAGSystem()
            assert rag_system is not None
            assert rag_system.session_store == {}
    
    def test_rag_system_query(self, mock_vector_store, mock_openai):
        """Test basic query processing."""
        with patch('os.path.exists', return_value=True):
            rag_system = RAGSystem()
            
            # Mock the chain response
            with patch.object(rag_system, 'rag_chain') as mock_chain:
                mock_chain.invoke.return_value = "Test response"
                
                response = rag_system.query("Test query", "test_session")
                assert response == "Test response"
                mock_chain.invoke.assert_called_once()
    
    def test_rag_system_retrieve(self, mock_vector_store, mock_openai):
        """Test document retrieval."""
        with patch('os.path.exists', return_value=True):
            rag_system = RAGSystem()
            
            result = rag_system.retrieve("water bottle")
            assert "documents" in result
            assert "products" in result
            assert "query_type" in result
            assert "context" in result
    
    def test_session_management(self, mock_vector_store, mock_openai):
        """Test session management functionality."""
        with patch('os.path.exists', return_value=True):
            rag_system = RAGSystem()
            
            # Test session info for non-existent session
            info = rag_system.get_session_info("nonexistent")
            assert info["exists"] is False
            assert info["message_count"] == 0
            
            # Test clearing non-existent session
            result = rag_system.clear_session("nonexistent")
            assert result is False


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_basic_workflow(self, mock_vector_store, mock_openai):
        """Test a complete workflow from query to response."""
        with patch('os.path.exists', return_value=True):
            rag_system = RAGSystem()
            
            # Test retrieval
            retrieval_result = rag_system.retrieve("recommend a water bottle")
            assert retrieval_result["query_type"] == QueryType.RECOMMENDATION
            assert len(retrieval_result["products"]) > 0
            
            # Test query processing
            with patch.object(rag_system, 'rag_chain') as mock_chain:
                mock_chain.invoke.return_value = "I recommend the HydroFlask 32oz..."
                
                response = rag_system.query("recommend a water bottle", "test_session")
                assert "HydroFlask" in response


def test_environment_variable_handling():
    """Test that environment variables are properly handled."""
    # Test with default values
    with patch.dict(os.environ, {}, clear=True):
        from chains import TOP_K, MODEL_CHAT, MODEL_EMBED
        # Should use defaults when env vars not set
        assert TOP_K == 5  # Default value
    
    # Test with custom values
    with patch.dict(os.environ, {
        'RAG_TOP_K_RESULTS': '10',
        'OPENAI_CHAT_MODEL': 'gpt-4',
        'OPENAI_EMBEDDING_MODEL': 'text-embedding-3-large'
    }):
        # Re-import to get updated values
        import importlib
        import chains
        importlib.reload(chains)
        
        assert chains.TOP_K == 10
        assert chains.MODEL_CHAT == 'gpt-4'
        assert chains.MODEL_EMBED == 'text-embedding-3-large'