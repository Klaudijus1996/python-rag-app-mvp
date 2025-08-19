import pytest
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app import app, get_rag_system, check_index_exists
from chains import RAGSystem
from schema import QueryType, ProductInfo


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for testing."""
    rag = Mock(spec=RAGSystem)
    rag.query.return_value = "This is a test response about storage containers."
    rag.retrieve.return_value = {
        "documents": [],
        "products": [
            ProductInfo(
                product_id="1",
                name="Storage Container Set",
                brand="TestBrand",
                category="Cleaning & Household",
                sub_category="Storage & Accessories",
                price=150.0,
                type="Storage Baskets",
                rating=4.2,
                description="A great storage container for testing"
            )
        ],
        "query_type": QueryType.RECOMMENDATION,
        "context": "Test context"
    }
    rag.get_session_info.return_value = {
        "session_id": "test_session",
        "message_count": 0,
        "exists": False
    }
    rag.clear_session.return_value = True
    return rag


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_with_index(self, client):
        """Test health endpoint when index exists."""
        with patch('app.check_index_exists', return_value=True):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["index_ready"] is True
            assert "embedding_model" in data
    
    def test_health_without_index(self, client):
        """Test health endpoint when index doesn't exist."""
        with patch('app.check_index_exists', return_value=False):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["index_ready"] is False


class TestChatEndpoint:
    """Test chat endpoint functionality."""
    
    def test_chat_success(self, client, mock_rag_system):
        """Test successful chat interaction."""
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            payload = {
                "session_id": "test_session",
                "query": "I need storage containers for my kitchen",
                "max_products": 3
            }
            response = client.post("/chat", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert "answer" in data
            assert "products" in data
            assert "session_id" in data
            assert data["session_id"] == "test_session"
            
            # Verify RAG system was called
            mock_rag_system.query.assert_called_once()
            mock_rag_system.retrieve.assert_called_once()
        finally:
            app.dependency_overrides.clear()
    
    def test_chat_empty_query(self, client, mock_rag_system):
        """Test chat with empty query."""
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            payload = {
                "session_id": "test_session",
                "query": "",
            }
            response = client.post("/chat", json=payload)
            assert response.status_code == 422  # Pydantic validation error
        finally:
            app.dependency_overrides.clear()
    
    def test_chat_missing_session_id(self, client, mock_rag_system):
        """Test chat without session ID."""
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            payload = {
                "query": "test query"
            }
            response = client.post("/chat", json=payload)
            assert response.status_code == 422  # Validation error
        finally:
            app.dependency_overrides.clear()
    
    def test_chat_with_filters(self, client, mock_rag_system):
        """Test chat with category and price filters."""
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            payload = {
                "session_id": "test_session",
                "query": "recommend a product",
                "category_filter": "Cleaning & Household",
                "price_range": {"min": 100.0, "max": 200.0},
                "max_products": 2
            }
            response = client.post("/chat", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert data["products"] is not None
        finally:
            app.dependency_overrides.clear()
    
    def test_chat_no_rag_system(self, client):
        """Test chat when RAG system is not available."""
        def failing_rag_system():
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG system not available. Please run ingestion first."
            )
        app.dependency_overrides[get_rag_system] = failing_rag_system
        try:
            payload = {
                "session_id": "test_session",
                "query": "test query"
            }
            response = client.post("/chat", json=payload)
            assert response.status_code == 503
        finally:
            app.dependency_overrides.clear()


class TestIngestEndpoint:
    """Test ingestion endpoint functionality."""
    
    @patch('app.ingest.load_and_process_data')
    @patch('app.ingest.chunk_documents')
    @patch('app.ingest.create_and_save_vector_store')
    def test_ingest_success(self, mock_create_and_save, mock_chunk, mock_load, client):
        """Test successful ingestion."""
        # Mock the ingestion process
        mock_docs = [Mock()]
        mock_chunks = [Mock(), Mock(), Mock()]
        mock_vector_store = Mock()
        mock_vector_store.store_type = "faiss"
        
        mock_load.return_value = mock_docs
        mock_chunk.return_value = mock_chunks
        mock_create_and_save.return_value = mock_vector_store
        
        with patch('app.check_index_exists', return_value=False):
            payload = {
                "force_reindex": False,
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
            response = client.post("/ingest", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "completed"
            assert data["chunks_indexed"] == 3
            assert data["products_processed"] == 1
            assert "processing_time_seconds" in data
    
    def test_ingest_already_exists(self, client):
        """Test ingestion when index already exists."""
        with patch('app.check_index_exists', return_value=True):
            payload = {"force_reindex": False}
            response = client.post("/ingest", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "already_exists"
    
    def test_ingest_force_reindex(self, client):
        """Test forced reindexing."""
        mock_vector_store = Mock()
        mock_vector_store.store_type = "faiss"
        
        with patch('app.check_index_exists', return_value=True), \
             patch('app.ingest.load_and_process_data', return_value=[]), \
             patch('app.ingest.chunk_documents', return_value=[]), \
             patch('app.ingest.create_and_save_vector_store', return_value=mock_vector_store):
            
            payload = {"force_reindex": True}
            response = client.post("/ingest", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "completed"
    
    @patch('app.ingest.load_and_process_data')
    def test_ingest_file_not_found(self, mock_load, client):
        """Test ingestion when data file is missing."""
        mock_load.side_effect = FileNotFoundError("Data file not found")
        
        with patch('app.check_index_exists', return_value=False):
            response = client.post("/ingest")
            assert response.status_code == 404


class TestSessionEndpoints:
    """Test session management endpoints."""
    
    def test_get_session_info(self, client, mock_rag_system):
        """Test getting session information."""
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            response = client.get("/sessions/test_session")
            assert response.status_code == 200
            
            data = response.json()
            assert data["session_id"] == "test_session"
            assert "message_count" in data
            assert "exists" in data
        finally:
            app.dependency_overrides.clear()
    
    def test_clear_session_success(self, client, mock_rag_system):
        """Test clearing a session successfully."""
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            response = client.delete("/sessions/test_session")
            assert response.status_code == 200
            
            data = response.json()
            assert "cleared successfully" in data["message"]
        finally:
            app.dependency_overrides.clear()
    
    def test_clear_session_not_found(self, client, mock_rag_system):
        """Test clearing a non-existent session."""
        mock_rag_system.clear_session.return_value = False
        
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            response = client.delete("/sessions/nonexistent")
            assert response.status_code == 200
            
            data = response.json()
            assert "not found" in data["message"]
        finally:
            app.dependency_overrides.clear()


class TestRetrieveEndpoint:
    """Test document retrieval endpoint."""
    
    def test_retrieve_documents(self, client, mock_rag_system):
        """Test document retrieval."""
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            response = client.get("/retrieve/storage containers")
            assert response.status_code == 200
            
            data = response.json()
            assert data["query"] == "storage containers"
            assert "query_type" in data
            assert "products" in data
            assert "document_count" in data
        finally:
            app.dependency_overrides.clear()


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_validation_error_response(self, client, mock_rag_system):
        """Test validation error handling."""
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            # Send invalid payload
            payload = {
                "session_id": "",  # Empty session_id should fail validation
                "query": "test"
            }
            response = client.post("/chat", json=payload)
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()
    
    def test_internal_server_error(self, client, mock_rag_system):
        """Test internal server error handling."""
        mock_rag_system.query.side_effect = Exception("Internal error")
        
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            payload = {
                "session_id": "test_session",
                "query": "test query"
            }
            response = client.post("/chat", json=payload)
            assert response.status_code == 500
        finally:
            app.dependency_overrides.clear()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_check_index_exists_true_with_vector_store(self):
        """Test index existence check using vector store abstraction."""
        with patch('vector_stores.VectorStoreFactory.create_from_env') as mock_factory:
            mock_vector_store = Mock()
            mock_vector_store.exists.return_value = True
            mock_factory.return_value = mock_vector_store
            
            assert check_index_exists() is True
            mock_vector_store.exists.assert_called_once()
    
    def test_check_index_exists_false_with_vector_store(self):
        """Test index existence check when vector store doesn't exist."""
        with patch('vector_stores.VectorStoreFactory.create_from_env') as mock_factory:
            mock_vector_store = Mock()
            mock_vector_store.exists.return_value = False
            mock_factory.return_value = mock_vector_store
            
            assert check_index_exists() is False
            mock_vector_store.exists.assert_called_once()
    
    def test_check_index_exists_fallback_to_path_check(self):
        """Test fallback to path checking when vector store creation fails."""
        with patch('vector_stores.VectorStoreFactory.create_from_env', side_effect=Exception("Factory error")), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.iterdir', return_value=['file1', 'file2']):
            
            assert check_index_exists() is True
    
    def test_check_index_exists_fallback_false(self):
        """Test fallback returns False when directory doesn't exist."""
        with patch('vector_stores.VectorStoreFactory.create_from_env', side_effect=Exception("Factory error")), \
             patch('pathlib.Path.exists', return_value=False):
            
            assert check_index_exists() is False


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "RAG MVP is running" in data["message"]


@pytest.mark.asyncio
class TestAsyncBehavior:
    """Test async behavior of the application."""
    
    async def test_async_chat_request(self, mock_rag_system):
        """Test async chat request handling."""
        from fastapi.testclient import TestClient
        
        app.dependency_overrides[get_rag_system] = lambda: mock_rag_system
        try:
            client = TestClient(app)
            payload = {
                "session_id": "async_test", 
                "query": "async test query"
            }
            response = client.post("/chat", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert data["session_id"] == "async_test"
        finally:
            app.dependency_overrides.clear()


class TestCORSMiddleware:
    """Test CORS middleware configuration."""
    
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set."""
        response = client.options("/chat")
        # Should not fail due to CORS
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled


class TestEnvironmentConfiguration:
    """Test environment-based configuration."""
    
    def test_app_configuration_with_env_vars(self):
        """Test app configuration with environment variables."""
        with patch.dict(os.environ, {
            'APP_ENV': 'production',
            'APP_HOST': '127.0.0.1',
            'APP_PORT': '9000'
        }):
            # Re-import app module to test env var handling
            import importlib
            import app
            importlib.reload(app)
            
            assert app.APP_ENV == 'production'
            assert app.APP_HOST == '127.0.0.1'
            assert app.APP_PORT == 9000