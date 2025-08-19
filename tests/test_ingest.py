"""Tests for the refactored ingestion pipeline with vector store abstraction."""

import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch

import pandas as pd
from langchain_core.documents import Document

import ingest


class TestIngestFunctions:
    """Test individual ingestion functions."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'index': [1, 2, 3],
            'product': ['Apple', 'Banana', 'Orange'],
            'brand': ['Fresh', 'Tropical', 'Citrus'],
            'category': ['Fruits', 'Fruits', 'Fruits'],
            'sub_category': ['Fresh Fruits', 'Tropical Fruits', 'Citrus Fruits'],
            'sale_price': [2.99, 1.49, 3.99],
            'type': ['Organic', 'Regular', 'Premium'],
            'rating': [4.5, 4.2, 4.8],
            'description': ['Fresh organic apple', 'Sweet banana', 'Juicy orange']
        })
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Name: Apple\nCategory: Fruits\nBrand: Fresh\nPrice: 2.99",
                metadata={
                    "product_id": "1",
                    "name": "Apple",
                    "brand": "Fresh",
                    "category": "Fruits",
                    "price": 2.99
                }
            ),
            Document(
                page_content="Name: Banana\nCategory: Fruits\nBrand: Tropical\nPrice: 1.49",
                metadata={
                    "product_id": "2",
                    "name": "Banana",
                    "brand": "Tropical",
                    "category": "Fruits",
                    "price": 1.49
                }
            )
        ]
    
    def test_row_to_text(self, sample_dataframe):
        """Test row to text conversion."""
        row = sample_dataframe.iloc[0]
        text = ingest.row_to_text(row)
        
        assert "Apple" in text
        assert "Fresh" in text
        assert "Fruits" in text
        assert "2.99" in text
        assert "Organic" in text
        assert "4.5" in text
    
    def test_create_document_metadata(self, sample_dataframe):
        """Test document metadata creation."""
        row = sample_dataframe.iloc[0]
        metadata = ingest.create_document_metadata(row)
        
        assert metadata["product_id"] == "1"
        assert metadata["name"] == "Apple"
        assert metadata["brand"] == "Fresh"
        assert metadata["category"] == "Fruits"
        assert metadata["price"] == 2.99
        assert metadata["type"] == "Organic"
        assert metadata["rating"] == 4.5
    
    @patch('ingest.Path.exists')
    @patch('pandas.read_csv')
    def test_load_and_process_data_success(self, mock_read_csv, mock_exists, sample_dataframe):
        """Test successful data loading and processing."""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_dataframe
        
        documents = ingest.load_and_process_data("test_file.csv")
        
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].metadata["name"] == "Apple"
        assert documents[1].metadata["name"] == "Banana"
        assert documents[2].metadata["name"] == "Orange"
    
    @patch('ingest.Path.exists')
    def test_load_and_process_data_file_not_found(self, mock_exists):
        """Test data loading when file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            ingest.load_and_process_data("nonexistent.csv")
    
    def test_chunk_documents(self, sample_documents):
        """Test document chunking."""
        chunks = ingest.chunk_documents(sample_documents)
        
        # Should return at least the original documents (might be more if chunked)
        assert len(chunks) >= len(sample_documents)
        assert all(isinstance(chunk, Document) for chunk in chunks)


class TestCreateAndSaveVectorStore:
    """Test the new create_and_save_vector_store function."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Product 1: Organic apples",
                metadata={"product_id": "1", "name": "Organic Apples", "price": 2.99}
            ),
            Document(
                page_content="Product 2: Fresh bananas",
                metadata={"product_id": "2", "name": "Fresh Bananas", "price": 1.49}
            )
        ]
    
    @patch('ingest.VectorStoreFactory.create_from_env')
    @patch('ingest.OpenAIEmbeddings')
    def test_create_and_save_vector_store_faiss(self, mock_embeddings_class, mock_factory, sample_documents):
        """Test create_and_save_vector_store with FAISS."""
        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.store_type = "faiss"
        mock_factory.return_value = mock_vector_store
        
        result = ingest.create_and_save_vector_store(sample_documents)
        
        # Verify factory was called
        mock_factory.assert_called_once()
        
        # Verify embeddings were created
        mock_embeddings_class.assert_called_once_with(model=ingest.EMBED_MODEL)
        
        # Verify vector store operations
        mock_vector_store.create_from_documents.assert_called_once_with(sample_documents, mock_embeddings)
        mock_vector_store.save.assert_called_once()
        
        assert result == mock_vector_store
    
    @patch('ingest.VectorStoreFactory.create_from_env')
    @patch('ingest.OpenAIEmbeddings')
    def test_create_and_save_vector_store_pinecone(self, mock_embeddings_class, mock_factory, sample_documents):
        """Test create_and_save_vector_store with Pinecone."""
        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.store_type = "pinecone"
        mock_factory.return_value = mock_vector_store
        
        result = ingest.create_and_save_vector_store(sample_documents)
        
        # Verify factory was called
        mock_factory.assert_called_once()
        
        # Verify vector store operations
        mock_vector_store.create_from_documents.assert_called_once_with(sample_documents, mock_embeddings)
        mock_vector_store.save.assert_called_once()  # Should be called even if it's a no-op for Pinecone
        
        assert result == mock_vector_store
        assert result.store_type == "pinecone"
    
    @patch('ingest.VectorStoreFactory.create_from_env')
    @patch('ingest.OpenAIEmbeddings')
    def test_create_and_save_vector_store_failure(self, mock_embeddings_class, mock_factory, sample_documents):
        """Test create_and_save_vector_store with factory failure."""
        mock_embeddings_class.return_value = Mock()
        mock_factory.side_effect = Exception("Factory creation failed")
        
        with pytest.raises(Exception, match="Factory creation failed"):
            ingest.create_and_save_vector_store(sample_documents)
    
    @patch('ingest.VectorStoreFactory.create_from_env')
    @patch('ingest.OpenAIEmbeddings')
    def test_create_and_save_vector_store_embeddings_failure(self, mock_embeddings_class, mock_factory, sample_documents):
        """Test create_and_save_vector_store with embeddings failure."""
        mock_embeddings_class.side_effect = Exception("Embeddings creation failed")
        mock_factory.return_value = Mock()
        
        with pytest.raises(Exception, match="Embeddings creation failed"):
            ingest.create_and_save_vector_store(sample_documents)


class TestMainIngestionPipeline:
    """Test the main ingestion pipeline."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Product 1: Test product",
                metadata={"product_id": "1", "name": "Test Product"}
            )
        ]
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            Document(
                page_content="Chunk 1: Test content",
                metadata={"product_id": "1", "chunk": "1"}
            ),
            Document(
                page_content="Chunk 2: More test content", 
                metadata={"product_id": "1", "chunk": "2"}
            )
        ]
    
    @patch('ingest.load_and_process_data')
    @patch('ingest.chunk_documents')
    @patch('ingest.create_and_save_vector_store')
    def test_main_success(self, mock_create_and_save, mock_chunk, mock_load, 
                         sample_documents, sample_chunks):
        """Test successful main pipeline execution."""
        # Mock the pipeline functions
        mock_load.return_value = sample_documents
        mock_chunk.return_value = sample_chunks
        mock_vector_store = Mock()
        mock_vector_store.store_type = "faiss"
        mock_create_and_save.return_value = mock_vector_store
        
        # Run main pipeline
        ingest.main()
        
        # Verify function calls
        mock_load.assert_called_once_with(ingest.DATA_PATH)
        mock_chunk.assert_called_once_with(sample_documents)
        mock_create_and_save.assert_called_once_with(sample_chunks)
    
    @patch('ingest.load_and_process_data')
    def test_main_no_documents(self, mock_load):
        """Test main pipeline when no documents are loaded."""
        mock_load.return_value = []
        
        # Should not raise an exception, just return early
        ingest.main()
        
        mock_load.assert_called_once()
    
    @patch('ingest.load_and_process_data')
    def test_main_file_not_found(self, mock_load):
        """Test main pipeline when data file is not found."""
        mock_load.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            ingest.main()
    
    @patch('ingest.load_and_process_data')
    @patch('ingest.chunk_documents')
    @patch('ingest.create_and_save_vector_store')
    def test_main_vector_store_failure(self, mock_create_and_save, mock_chunk, mock_load,
                                      sample_documents, sample_chunks):
        """Test main pipeline when vector store creation fails."""
        mock_load.return_value = sample_documents
        mock_chunk.return_value = sample_chunks
        mock_create_and_save.side_effect = Exception("Vector store creation failed")
        
        with pytest.raises(Exception, match="Vector store creation failed"):
            ingest.main()


class TestEnvironmentConfiguration:
    """Test environment-based configuration in ingestion."""
    
    def test_default_configuration_values(self):
        """Test that default configuration values are set correctly."""
        assert ingest.DATA_PATH == "data/big-basket-products-20.csv"
        assert ingest.EMBED_MODEL == os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        assert ingest.CHUNK_SIZE == int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        assert ingest.CHUNK_OVERLAP == int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    
    def test_environment_variable_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        monkeypatch.setenv("RAG_CHUNK_SIZE", "1500")
        monkeypatch.setenv("RAG_CHUNK_OVERLAP", "300")
        
        # Re-import to get updated values
        import importlib
        importlib.reload(ingest)
        
        assert ingest.EMBED_MODEL == "text-embedding-ada-002"
        assert ingest.CHUNK_SIZE == 1500
        assert ingest.CHUNK_OVERLAP == 300


class TestBackwardCompatibility:
    """Test backward compatibility with old ingestion approach."""
    
    @patch('ingest.VectorStoreFactory.create_from_env')
    def test_faiss_default_behavior(self, mock_factory, monkeypatch):
        """Test that FAISS remains the default vector store."""
        # Don't set VECTOR_STORE_TYPE, should default to FAISS
        monkeypatch.delenv("VECTOR_STORE_TYPE", raising=False)
        
        mock_vector_store = Mock()
        mock_vector_store.store_type = "faiss"
        mock_factory.return_value = mock_vector_store
        
        with patch('ingest.OpenAIEmbeddings'):
            store = ingest.create_and_save_vector_store([])
            
        assert store.store_type == "faiss"
    
    @patch('ingest.VectorStoreFactory.create_from_env')
    def test_faiss_explicit_configuration(self, mock_factory, monkeypatch):
        """Test explicit FAISS configuration."""
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("FAISS_INDEX_DIR", "custom_faiss_index")
        
        mock_vector_store = Mock()
        mock_vector_store.store_type = "faiss"
        mock_factory.return_value = mock_vector_store
        
        with patch('ingest.OpenAIEmbeddings'):
            store = ingest.create_and_save_vector_store([])
            
        assert store.store_type == "faiss"


class TestErrorHandling:
    """Test error handling in ingestion pipeline."""
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted data rows."""
        corrupted_df = pd.DataFrame({
            'index': [1, 2, None],
            'product': ['Apple', None, 'Orange'],
            'brand': ['Fresh', 'Missing', ''],
            'category': ['Fruits', None, 'Fruits'],
            'sale_price': [2.99, 'invalid', 3.99],
            'description': ['Good apple', '', 'Sweet orange']
        })
        
        with patch('ingest.Path.exists', return_value=True), \
             patch('pandas.read_csv', return_value=corrupted_df):
            
            documents = ingest.load_and_process_data("test.csv")
            
            # Should still create documents, but handle corrupted data gracefully
            assert len(documents) <= 3  # Some rows might be skipped
            
            # Valid documents should still be created
            valid_docs = [doc for doc in documents if doc.metadata.get("name")]
            assert len(valid_docs) >= 1
    
    @patch('ingest.VectorStoreFactory.create_from_env')
    def test_missing_vector_store_dependencies(self, mock_factory):
        """Test handling when vector store dependencies are missing."""
        mock_factory.side_effect = ImportError("Missing dependencies")
        
        with pytest.raises(ImportError):
            ingest.create_and_save_vector_store([])
    
    def test_empty_documents_list(self):
        """Test handling of empty documents list."""
        with patch('ingest.VectorStoreFactory.create_from_env') as mock_factory, \
             patch('ingest.OpenAIEmbeddings'):
            
            mock_vector_store = Mock()
            mock_factory.return_value = mock_vector_store
            
            # Should handle empty list gracefully
            result = ingest.create_and_save_vector_store([])
            
            # The embeddings should be passed, not the factory
            mock_vector_store.create_from_documents.assert_called_once()
            assert result == mock_vector_store


class TestIntegrationWithVectorStores:
    """Integration tests with actual vector store implementations."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Organic apple from local farm",
                metadata={"product_id": "1", "name": "Organic Apple", "price": 2.99}
            )
        ]
    
    def test_integration_with_faiss(self, temp_dir, sample_documents, monkeypatch):
        """Test integration with FAISS vector store."""
        # Set up FAISS environment
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("FAISS_INDEX_DIR", temp_dir)
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        with patch('vector_stores.factory.OpenAIEmbeddings') as mock_embeddings_class, \
             patch('vector_stores.faiss_store.FAISS') as mock_faiss_class:
            
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            mock_faiss_instance = Mock()
            mock_faiss_class.from_documents.return_value = mock_faiss_instance
            
            # Test the integration  
            result = ingest.create_and_save_vector_store(sample_documents)
            
            assert result.store_type == "faiss"
            # The actual calls happen on the mock store created by the factory
            mock_embeddings_class.assert_called_once()
            mock_faiss_class.from_documents.assert_called_once()
    
    @patch('vector_stores.pinecone_store.PINECONE_AVAILABLE', True)
    def test_integration_with_pinecone(self, sample_documents, monkeypatch):
        """Test integration with Pinecone vector store."""
        # Set up Pinecone environment
        monkeypatch.setenv("VECTOR_STORE_TYPE", "pinecone")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
        monkeypatch.setenv("PINECONE_REGION", "us-east-1")
        monkeypatch.setenv("PINECONE_CLOUD", "aws")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        with patch('vector_stores.factory.OpenAIEmbeddings') as mock_embeddings_class, \
             patch('vector_stores.pinecone_store.Pinecone') as mock_pinecone_class, \
             patch('vector_stores.pinecone_store.PineconeVectorStore') as mock_pinecone_vector_store:
            
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            mock_pinecone_instance = Mock()
            mock_pinecone_class.return_value = mock_pinecone_instance
            mock_pinecone_instance.list_indexes.return_value.indexes = []
            
            mock_vector_store_instance = Mock()
            mock_pinecone_vector_store.from_documents.return_value = mock_vector_store_instance
            
            # Test the integration
            with patch('vector_stores.pinecone_store.PineconeVectorStoreWrapper._wait_for_index_ready'):
                result = ingest.create_and_save_vector_store(sample_documents)
                
                assert result.store_type == "pinecone"
                # The actual calls happen on the mock store created by the factory
                mock_embeddings_class.assert_called_once()
                mock_pinecone_vector_store.from_documents.assert_called_once()
    
    def test_switching_between_vector_stores(self, temp_dir, sample_documents, monkeypatch):
        """Test switching between different vector stores via environment."""
        with patch('vector_stores.factory.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            # Test FAISS first
            monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
            monkeypatch.setenv("FAISS_INDEX_DIR", temp_dir)
            
            with patch('vector_stores.faiss_store.FAISS'):
                faiss_result = ingest.create_and_save_vector_store(sample_documents)
                assert faiss_result.store_type == "faiss"
            
            # Switch to Pinecone
            monkeypatch.setenv("VECTOR_STORE_TYPE", "pinecone")
            monkeypatch.setenv("PINECONE_API_KEY", "test-key")
            monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
            monkeypatch.setenv("PINECONE_REGION", "us-east-1")
            monkeypatch.setenv("PINECONE_CLOUD", "aws")
            
            with patch('vector_stores.pinecone_store.PINECONE_AVAILABLE', True), \
                 patch('vector_stores.pinecone_store.Pinecone'), \
                 patch('vector_stores.pinecone_store.PineconeVectorStore'), \
                 patch('vector_stores.pinecone_store.PineconeVectorStoreWrapper._wait_for_index_ready'):
                
                pinecone_result = ingest.create_and_save_vector_store(sample_documents)
                assert pinecone_result.store_type == "pinecone"