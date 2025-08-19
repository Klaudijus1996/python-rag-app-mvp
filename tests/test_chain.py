import pytest
import os
from unittest.mock import Mock, patch

from chains import RAGSystem, detect_query_type, format_docs, extract_products_from_docs
from schema import QueryType, ProductInfo
from langchain_core.documents import Document


class TestQueryTypeDetection:
    """Test query type detection functionality."""
    
    def test_recommendation_detection(self):
        queries = [
            "I need good hand soap",
            "Can you recommend storage containers?",
            "What's the best cleaning wipes under 200?",
            "Help me find household products",
            "Looking for organic products"
        ]
        for query in queries:
            assert detect_query_type(query) == QueryType.RECOMMENDATION
    
    def test_comparison_detection(self):
        queries = [
            "Compare Nivea vs other soap brands",
            "What's the difference between these cleaning wipes?",
            "Which is better: container A or container B?",
            "Brand vs generic comparison",
            "Show me the differences between these products"
        ]
        for query in queries:
            assert detect_query_type(query) == QueryType.COMPARISON
    
    def test_complement_detection(self):
        queries = [
            "What cleaning supplies go with these wipes?",
            "What pairs well with this storage container?",
            "Compatible products for household cleaning",
            "What works with this soap?",
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
                page_content="Great storage container for kitchen",
                metadata={
                    "name": "Storage Container Set",
                    "brand": "Nakoda",
                    "price": 149.0,
                    "category": "Cleaning & Household",
                    "sub_category": "Bins & Bathroom Ware",
                    "type": "Storage Baskets",
                    "rating": 3.7,
                    "product_id": "4"
                }
            ),
            Document(
                page_content="Multipurpose cleaning wipes",
                metadata={
                    "name": "Germ Removal Wipes",
                    "brand": "Nature Protect",
                    "price": 169.0,
                    "category": "Cleaning & Household",
                    "sub_category": "All Purpose Cleaners",
                    "type": "Disinfectant Spray & Cleaners",
                    "rating": 3.3,
                    "product_id": "6"
                }
            )
        ]
        
        result = format_docs(docs)
        assert "Storage Container Set" in result
        assert "Nakoda" in result
        assert "149.0" in result
        assert "Germ Removal Wipes" in result
        assert "Nature Protect" in result
        assert "3.7" in result


class TestProductExtraction:
    """Test product information extraction."""
    
    def test_extract_products_from_docs(self):
        docs = [
            Document(
                page_content="High-quality storage container with excellent design",
                metadata={
                    "product_id": "4",
                    "name": "Storage Container Set",
                    "brand": "Nakoda",
                    "category": "Cleaning & Household",
                    "sub_category": "Bins & Bathroom Ware",
                    "price": 149.0,
                    "type": "Storage Baskets",
                    "rating": 3.7
                }
            )
        ]
        
        products = extract_products_from_docs(docs)
        assert len(products) == 1
        
        product = products[0]
        assert isinstance(product, ProductInfo)
        assert product.product_id == "4"
        assert product.name == "Storage Container Set"
        assert product.brand == "Nakoda"
        assert product.category == "Cleaning & Household"
        assert product.sub_category == "Bins & Bathroom Ware"
        assert product.price == 149.0
        assert product.type == "Storage Baskets"
        assert product.rating == 3.7
    
    def test_extract_products_with_invalid_data(self):
        docs = [
            Document(
                page_content="Test content",
                metadata={
                    "product_id": "1",
                    "name": "Test Product",
                    "brand": "Test Brand",
                    "category": "Test Category",
                    "price": "invalid_price"  # Invalid price
                }
            )
        ]
        
        products = extract_products_from_docs(docs)
        # Should handle invalid data gracefully and create product with default values
        assert len(products) == 1
        product = products[0]
        assert product.price == 0.0  # Invalid price converted to default


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing using the new abstraction."""
    with patch('chains.VectorStoreFactory.create_from_env') as mock_factory:
        mock_store = Mock()
        mock_retriever = Mock()
        
        # Mock documents to return
        mock_docs = [
            Document(
                page_content="Excellent storage container for household use",
                metadata={
                    "product_id": "4",
                    "name": "Storage Container Set",
                    "brand": "Nakoda",
                    "category": "Cleaning & Household",
                    "sub_category": "Bins & Bathroom Ware",
                    "price": 149.0,
                    "type": "Storage Baskets",
                    "rating": 3.7
                }
            )
        ]
        
        mock_retriever.get_relevant_documents.return_value = mock_docs
        mock_retriever.invoke.return_value = mock_docs
        mock_store.get_retriever.return_value = mock_retriever
        mock_store.load_existing.return_value = None
        mock_store.store_type = "faiss"
        mock_factory.return_value = mock_store
        
        yield mock_store


@pytest.fixture
def mock_openai():
    """Mock OpenAI components."""
    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings, \
         patch('chains.ChatOpenAI') as mock_chat:
        
        mock_embeddings.return_value = Mock()
        mock_chat_instance = Mock()
        mock_chat_instance.invoke.return_value = "This is a test response"
        mock_chat.return_value = mock_chat_instance
        
        yield mock_embeddings, mock_chat


class TestRAGSystem:
    """Test the complete RAG system."""
    
    def test_rag_system_initialization(self, mock_vector_store, mock_openai, monkeypatch):
        """Test RAG system can be initialized."""
        # Set up environment variables for vector store configuration
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("FAISS_INDEX_DIR", "test_index")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        rag_system = RAGSystem()
        assert rag_system is not None
        assert rag_system.session_store == {}
    
    def test_rag_system_query(self, mock_vector_store, mock_openai, monkeypatch):
        """Test basic query processing."""
        # Set up environment variables for vector store configuration
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("FAISS_INDEX_DIR", "test_index")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        rag_system = RAGSystem()
        
        # Mock the chain response
        with patch.object(rag_system, 'rag_chain') as mock_chain:
            mock_chain.invoke.return_value = "Test response"
            
            response = rag_system.query("Test query", "test_session")
            assert response == "Test response"
            mock_chain.invoke.assert_called_once()
    
    def test_rag_system_retrieve(self, mock_vector_store, mock_openai, monkeypatch):
        """Test document retrieval."""
        # Set up environment variables for vector store configuration
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("FAISS_INDEX_DIR", "test_index")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        rag_system = RAGSystem()
        
        result = rag_system.retrieve("storage containers")
        assert "documents" in result
        assert "products" in result
        assert "query_type" in result
        assert "context" in result
    
    def test_session_management(self, mock_vector_store, mock_openai, monkeypatch):
        """Test session management functionality."""
        # Set up environment variables for vector store configuration
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("FAISS_INDEX_DIR", "test_index")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
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
    async def test_basic_workflow(self, mock_vector_store, mock_openai, monkeypatch):
        """Test a complete workflow from query to response."""
        # Set up environment variables for vector store configuration
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("FAISS_INDEX_DIR", "test_index")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        rag_system = RAGSystem()
        
        # Test retrieval
        retrieval_result = rag_system.retrieve("recommend storage containers")
        assert retrieval_result["query_type"] == QueryType.RECOMMENDATION
        assert len(retrieval_result["products"]) > 0
        
        # Test query processing
        with patch.object(rag_system, 'rag_chain') as mock_chain:
            mock_chain.invoke.return_value = "I recommend the Storage Container Set..."
            
            response = rag_system.query("recommend storage containers", "test_session")
            assert "Storage Container" in response


class TestLoadRetriever:
    """Test the load_retriever function with vector store abstraction."""
    
    def test_load_retriever_with_factory(self, mock_vector_store, monkeypatch):
        """Test load_retriever uses the factory pattern correctly."""
        from chains import load_retriever
        
        # Set up environment variables for vector store configuration
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("FAISS_INDEX_DIR", "test_index")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        retriever = load_retriever(k=10, search_type="mmr")
        
        # Verify that the vector store was created and loaded
        mock_vector_store.load_existing.assert_called_once()
        mock_vector_store.get_retriever.assert_called_once_with(
            search_type="mmr",
            k=10,
            lambda_mult=0.4
        )
        assert retriever is not None
    
    def test_load_retriever_with_similarity_search(self, mock_vector_store, monkeypatch):
        """Test load_retriever with similarity search."""
        from chains import load_retriever
        
        # Set up environment variables for vector store configuration
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("FAISS_INDEX_DIR", "test_index")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        retriever = load_retriever(k=5, search_type="similarity")
        
        # Verify the correct search type was used
        mock_vector_store.get_retriever.assert_called_once_with(
            search_type="similarity",
            k=5,
            lambda_mult=None
        )
        assert retriever is not None


def test_environment_variable_handling():
    """Test that environment variables are properly handled."""
    # Test with default values
    with patch.dict(os.environ, {}, clear=True):
        from chains import TOP_K
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