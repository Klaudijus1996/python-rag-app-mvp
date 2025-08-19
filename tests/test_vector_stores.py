"""Tests for vector store implementations."""

import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from vector_stores.base import VectorStoreInterface
from vector_stores.faiss_store import FAISSVectorStore
from vector_stores.factory import VectorStoreFactory
from vector_stores.config import VectorStoreConfig, FAISSConfig, PineconeConfig


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig."""

    def test_faiss_config_from_env(self, monkeypatch):
        """Test creating FAISS config from environment variables."""
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("FAISS_INDEX_DIR", "test_faiss_index")

        config = VectorStoreConfig.from_env()

        assert config.store_type == "faiss"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.faiss_config is not None
        assert config.faiss_config.index_dir == "test_faiss_index"
        assert config.pinecone_config is None

    def test_pinecone_config_from_env(self, monkeypatch):
        """Test creating Pinecone config from environment variables."""
        monkeypatch.setenv("VECTOR_STORE_TYPE", "pinecone")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("PINECONE_API_KEY", "test-api-key")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
        monkeypatch.setenv("PINECONE_REGION", "us-east-1")
        monkeypatch.setenv("PINECONE_CLOUD", "aws")
        monkeypatch.setenv("PINECONE_DIMENSION", "1536")
        monkeypatch.setenv("PINECONE_METRIC", "cosine")

        config = VectorStoreConfig.from_env()

        assert config.store_type == "pinecone"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.pinecone_config is not None
        assert config.pinecone_config.api_key == "test-api-key"
        assert config.pinecone_config.index_name == "test-index"
        assert config.pinecone_config.region == "us-east-1"
        assert config.pinecone_config.cloud == "aws"
        assert config.pinecone_config.dimension == 1536
        assert config.pinecone_config.metric == "cosine"
        assert config.faiss_config is None

    def test_missing_required_config_raises_error(self, monkeypatch):
        """Test that missing required config raises ValueError."""
        # Clear any existing Pinecone environment variables first
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        monkeypatch.delenv("PINECONE_REGION", raising=False)
        monkeypatch.delenv("PINECONE_CLOUD", raising=False)

        monkeypatch.setenv("VECTOR_STORE_TYPE", "pinecone")
        # Missing PINECONE_API_KEY (other required fields will be empty too)

        with pytest.raises(ValueError, match="PINECONE_API_KEY is required"):
            VectorStoreConfig.from_env()

    def test_default_values(self, monkeypatch):
        """Test default configuration values."""
        monkeypatch.setenv("FAISS_INDEX_DIR", "test_index")
        monkeypatch.delenv("VECTOR_STORE_TYPE", raising=False)

        config = VectorStoreConfig.from_env()

        # Should default to FAISS
        assert config.store_type == "faiss"
        assert config.embedding_model == "text-embedding-3-small"  # Default
        assert config.faiss_config.index_dir == "test_index"


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        return Mock(spec=OpenAIEmbeddings)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Product 1: Organic apples",
                metadata={"product_id": "1", "name": "Organic Apples", "price": 2.99},
            ),
            Document(
                page_content="Product 2: Fresh bananas",
                metadata={"product_id": "2", "name": "Fresh Bananas", "price": 1.49},
            ),
        ]

    def test_faiss_store_initialization(self, temp_dir, mock_embeddings):
        """Test FAISS store initialization."""
        store = FAISSVectorStore(temp_dir, mock_embeddings)

        assert store.index_dir == temp_dir
        assert store.embeddings == mock_embeddings
        assert store.vector_store is None
        assert store.store_type == "faiss"

    @patch("vector_stores.faiss_store.FAISS")
    def test_create_from_documents(
        self, mock_faiss_class, temp_dir, mock_embeddings, sample_documents
    ):
        """Test creating FAISS store from documents."""
        mock_faiss_instance = Mock()
        mock_faiss_class.from_documents.return_value = mock_faiss_instance

        store = FAISSVectorStore(temp_dir, mock_embeddings)
        store.create_from_documents(sample_documents, mock_embeddings)

        mock_faiss_class.from_documents.assert_called_once_with(
            sample_documents, mock_embeddings
        )
        assert store.vector_store == mock_faiss_instance

    def test_exists_false_when_directory_missing(self, temp_dir, mock_embeddings):
        """Test exists returns False when directory doesn't exist."""
        non_existent_dir = os.path.join(temp_dir, "non_existent")
        store = FAISSVectorStore(non_existent_dir, mock_embeddings)

        assert store.exists() is False

    def test_exists_false_when_directory_empty(self, temp_dir, mock_embeddings):
        """Test exists returns False when directory is empty."""
        store = FAISSVectorStore(temp_dir, mock_embeddings)

        assert store.exists() is False

    def test_exists_true_when_files_present(self, temp_dir, mock_embeddings):
        """Test exists returns True when files are present."""
        # Create a dummy file in the temp directory
        with open(os.path.join(temp_dir, "index.faiss"), "w") as f:
            f.write("dummy")

        store = FAISSVectorStore(temp_dir, mock_embeddings)

        assert store.exists() is True

    @patch("vector_stores.faiss_store.FAISS")
    def test_load_existing_success(self, mock_faiss_class, temp_dir, mock_embeddings):
        """Test loading existing FAISS store."""
        # Create a dummy file to simulate existing store
        with open(os.path.join(temp_dir, "index.faiss"), "w") as f:
            f.write("dummy")

        mock_faiss_instance = Mock()
        mock_faiss_class.load_local.return_value = mock_faiss_instance

        store = FAISSVectorStore(temp_dir, mock_embeddings)
        store.load_existing()

        mock_faiss_class.load_local.assert_called_once_with(
            temp_dir, mock_embeddings, allow_dangerous_deserialization=True
        )
        assert store.vector_store == mock_faiss_instance

    def test_load_existing_file_not_found(self, temp_dir, mock_embeddings):
        """Test loading existing FAISS store when files don't exist."""
        store = FAISSVectorStore(temp_dir, mock_embeddings)

        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            store.load_existing()

    def test_save_success(self, temp_dir, mock_embeddings):
        """Test saving FAISS store."""
        mock_vector_store = Mock()

        store = FAISSVectorStore(temp_dir, mock_embeddings)
        store.vector_store = mock_vector_store

        store.save()

        mock_vector_store.save_local.assert_called_once_with(temp_dir)

    def test_save_no_vector_store_raises_error(self, temp_dir, mock_embeddings):
        """Test saving when no vector store is initialized."""
        store = FAISSVectorStore(temp_dir, mock_embeddings)

        with pytest.raises(RuntimeError, match="No vector store to save"):
            store.save()

    def test_get_retriever_mmr(self, temp_dir, mock_embeddings):
        """Test getting MMR retriever."""
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever

        store = FAISSVectorStore(temp_dir, mock_embeddings)
        store.vector_store = mock_vector_store

        retriever = store.get_retriever(search_type="mmr", k=10)

        assert retriever == mock_retriever
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 20,  # min(20, 4 * k) where k=10 gives 20
                "lambda_mult": 0.4,
            },
        )

    def test_get_retriever_similarity(self, temp_dir, mock_embeddings):
        """Test getting similarity retriever."""
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever

        store = FAISSVectorStore(temp_dir, mock_embeddings)
        store.vector_store = mock_vector_store

        retriever = store.get_retriever(search_type="similarity", k=5)

        assert retriever == mock_retriever
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 5}
        )

    def test_get_retriever_no_vector_store_raises_error(
        self, temp_dir, mock_embeddings
    ):
        """Test getting retriever when no vector store is initialized."""
        store = FAISSVectorStore(temp_dir, mock_embeddings)

        with pytest.raises(RuntimeError, match="Vector store not initialized"):
            store.get_retriever()

    def test_delete_success(self, temp_dir, mock_embeddings):
        """Test deleting FAISS store files."""
        # Create dummy files
        index_file = os.path.join(temp_dir, "index.faiss")
        pkl_file = os.path.join(temp_dir, "index.pkl")

        with open(index_file, "w") as f:
            f.write("dummy")
        with open(pkl_file, "w") as f:
            f.write("dummy")

        store = FAISSVectorStore(temp_dir, mock_embeddings)
        store.delete()

        assert not os.path.exists(index_file)
        assert not os.path.exists(pkl_file)


class TestPineconeVectorStore:
    """Tests for PineconeVectorStoreWrapper."""

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        return Mock(spec=OpenAIEmbeddings)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Product 1: Organic apples",
                metadata={"product_id": "1", "name": "Organic Apples", "price": 2.99},
            )
        ]

    @pytest.fixture
    def pinecone_config(self):
        """Create Pinecone configuration."""
        return {
            "api_key": "test-api-key",
            "region": "us-east-1",
            "cloud": "aws",
            "dimension": 1536,
            "metric": "cosine",
        }

    @patch("vector_stores.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("vector_stores.pinecone_store.Pinecone")
    def test_pinecone_store_initialization(
        self, mock_pinecone_class, mock_embeddings, pinecone_config
    ):
        """Test Pinecone store initialization."""
        from vector_stores.pinecone_store import PineconeVectorStoreWrapper

        mock_pinecone_instance = Mock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        store = PineconeVectorStoreWrapper(
            "test-index", mock_embeddings, pinecone_config
        )

        assert store.index_name == "test-index"
        assert store.embeddings == mock_embeddings
        assert store.api_key == "test-api-key"
        assert store.region == "us-east-1"
        assert store.cloud == "aws"
        assert store.dimension == 1536
        assert store.metric == "cosine"
        assert store.store_type == "pinecone"

        mock_pinecone_class.assert_called_once_with(api_key="test-api-key")

    @patch("vector_stores.pinecone_store.PINECONE_AVAILABLE", False)
    def test_pinecone_store_unavailable(self, mock_embeddings, pinecone_config):
        """Test Pinecone store when dependencies are not available."""
        from vector_stores.pinecone_store import PineconeVectorStoreWrapper

        with pytest.raises(ImportError, match="Pinecone dependencies not available"):
            PineconeVectorStoreWrapper("test-index", mock_embeddings, pinecone_config)

    @patch("vector_stores.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("vector_stores.pinecone_store.Pinecone")
    @patch("vector_stores.pinecone_store.PineconeVectorStore")
    def test_create_from_documents(
        self,
        mock_pinecone_vector_store,
        mock_pinecone_class,
        mock_embeddings,
        pinecone_config,
        sample_documents,
    ):
        """Test creating Pinecone store from documents."""
        from vector_stores.pinecone_store import PineconeVectorStoreWrapper

        mock_pinecone_instance = Mock()
        mock_pinecone_class.return_value = mock_pinecone_instance
        mock_pinecone_instance.list_indexes.return_value.indexes = []  # Index doesn't exist

        mock_vector_store_instance = Mock()
        mock_pinecone_vector_store.from_documents.return_value = (
            mock_vector_store_instance
        )

        store = PineconeVectorStoreWrapper(
            "test-index", mock_embeddings, pinecone_config
        )

        with patch.object(store, "_wait_for_index_ready"):
            store.create_from_documents(sample_documents, mock_embeddings)

        # Verify index creation
        mock_pinecone_instance.create_index.assert_called_once()

        # Verify vector store creation
        mock_pinecone_vector_store.from_documents.assert_called_once_with(
            documents=sample_documents,
            embedding=mock_embeddings,
            index_name="test-index",
            pinecone_api_key="test-api-key",
        )

        assert store.vector_store == mock_vector_store_instance

    @patch("vector_stores.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("vector_stores.pinecone_store.Pinecone")
    def test_exists_true(self, mock_pinecone_class, mock_embeddings, pinecone_config):
        """Test exists returns True when index exists."""
        from vector_stores.pinecone_store import PineconeVectorStoreWrapper

        mock_pinecone_instance = Mock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        mock_index = Mock()
        mock_index.name = "test-index"
        mock_pinecone_instance.list_indexes.return_value.indexes = [mock_index]

        store = PineconeVectorStoreWrapper(
            "test-index", mock_embeddings, pinecone_config
        )

        assert store.exists() is True

    @patch("vector_stores.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("vector_stores.pinecone_store.Pinecone")
    def test_exists_false(self, mock_pinecone_class, mock_embeddings, pinecone_config):
        """Test exists returns False when index doesn't exist."""
        from vector_stores.pinecone_store import PineconeVectorStoreWrapper

        mock_pinecone_instance = Mock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        mock_pinecone_instance.list_indexes.return_value.indexes = []

        store = PineconeVectorStoreWrapper(
            "test-index", mock_embeddings, pinecone_config
        )

        assert store.exists() is False

    @patch("vector_stores.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("vector_stores.pinecone_store.Pinecone")
    def test_delete_success(
        self, mock_pinecone_class, mock_embeddings, pinecone_config
    ):
        """Test deleting Pinecone index."""
        from vector_stores.pinecone_store import PineconeVectorStoreWrapper

        mock_pinecone_instance = Mock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        mock_index = Mock()
        mock_index.name = "test-index"
        mock_pinecone_instance.list_indexes.return_value.indexes = [mock_index]

        store = PineconeVectorStoreWrapper(
            "test-index", mock_embeddings, pinecone_config
        )
        store.delete()

        mock_pinecone_instance.delete_index.assert_called_once_with("test-index")

    @patch("vector_stores.pinecone_store.PINECONE_AVAILABLE", True)
    @patch("vector_stores.pinecone_store.Pinecone")
    def test_region_and_cloud_configuration(self, mock_pinecone_class, mock_embeddings):
        """Test that region and cloud are configured correctly."""
        from vector_stores.pinecone_store import PineconeVectorStoreWrapper

        config = {
            "api_key": "test-key",
            "region": "us-west-2",
            "cloud": "aws",
            "dimension": 1536,
            "metric": "cosine",
        }

        mock_pinecone_instance = Mock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        store = PineconeVectorStoreWrapper("test-index", mock_embeddings, config)

        assert store.region == "us-west-2"
        assert store.cloud == "aws"


class TestVectorStoreFactory:
    """Tests for VectorStoreFactory."""

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        return Mock(spec=OpenAIEmbeddings)

    @patch("vector_stores.factory.OpenAIEmbeddings")
    def test_create_faiss_store(self, mock_embeddings_class, mock_embeddings):
        """Test creating FAISS store via factory."""
        mock_embeddings_class.return_value = mock_embeddings

        config = VectorStoreConfig(
            store_type="faiss",
            embedding_model="text-embedding-3-small",
            faiss_config=FAISSConfig(index_dir="test_index"),
        )

        with patch("vector_stores.factory.FAISSVectorStore") as mock_faiss_class:
            mock_faiss_instance = Mock()
            mock_faiss_class.return_value = mock_faiss_instance

            result = VectorStoreFactory.create_vector_store(config)

            mock_faiss_class.assert_called_once_with(
                index_dir="test_index", embeddings=mock_embeddings
            )
            assert result == mock_faiss_instance

    @patch("vector_stores.factory.OpenAIEmbeddings")
    def test_create_pinecone_store(self, mock_embeddings_class, mock_embeddings):
        """Test creating Pinecone store via factory."""
        mock_embeddings_class.return_value = mock_embeddings

        config = VectorStoreConfig(
            store_type="pinecone",
            embedding_model="text-embedding-3-small",
            pinecone_config=PineconeConfig(
                api_key="test-key",
                index_name="test-index",
                region="us-east-1",
                cloud="aws",
                dimension=1536,
                metric="cosine",
            ),
        )

        with patch(
            "vector_stores.factory.PineconeVectorStoreWrapper"
        ) as mock_pinecone_class:
            mock_pinecone_instance = Mock()
            mock_pinecone_class.return_value = mock_pinecone_instance

            result = VectorStoreFactory.create_vector_store(config)

            expected_config = {
                "api_key": "test-key",
                "region": "us-east-1",
                "cloud": "aws",
                "dimension": 1536,
                "metric": "cosine",
            }

            mock_pinecone_class.assert_called_once_with(
                index_name="test-index",
                embeddings=mock_embeddings,
                config=expected_config,
            )
            assert result == mock_pinecone_instance

    def test_unsupported_store_type_raises_error(self):
        """Test that unsupported store type raises ValueError."""
        config = VectorStoreConfig(
            store_type="unsupported", embedding_model="text-embedding-3-small"
        )

        with pytest.raises(
            ValueError, match="Unsupported vector store type: unsupported"
        ):
            VectorStoreFactory.create_vector_store(config)

    def test_faiss_missing_config_raises_error(self):
        """Test that FAISS store without config raises ValueError."""
        config = VectorStoreConfig(
            store_type="faiss", embedding_model="text-embedding-3-small"
        )

        with pytest.raises(ValueError, match="FAISS configuration is required"):
            VectorStoreFactory.create_vector_store(config)

    def test_pinecone_missing_config_raises_error(self):
        """Test that Pinecone store without config raises ValueError."""
        config = VectorStoreConfig(
            store_type="pinecone", embedding_model="text-embedding-3-small"
        )

        with pytest.raises(ValueError, match="Pinecone configuration is required"):
            VectorStoreFactory.create_vector_store(config)

    @patch("vector_stores.factory.VectorStoreConfig.from_env")
    @patch("vector_stores.factory.VectorStoreFactory.create_vector_store")
    def test_create_from_env(self, mock_create_store, mock_from_env):
        """Test creating store from environment variables."""
        mock_config = Mock()
        mock_from_env.return_value = mock_config
        mock_store = Mock()
        mock_create_store.return_value = mock_store

        result = VectorStoreFactory.create_from_env()

        mock_from_env.assert_called_once()
        mock_create_store.assert_called_once_with(mock_config)
        assert result == mock_store

    def test_get_supported_types(self):
        """Test getting supported store types."""
        types = VectorStoreFactory.get_supported_types()

        assert isinstance(types, list)
        assert "faiss" in types
        assert "pinecone" in types
        assert len(types) == 2


class TestVectorStoreInterface:
    """Tests for VectorStoreInterface abstract base class."""

    def test_cannot_instantiate_interface_directly(self):
        """Test that abstract interface cannot be instantiated."""
        with pytest.raises(TypeError):
            VectorStoreInterface()

    def test_interface_defines_required_methods(self):
        """Test that interface defines all required abstract methods."""
        required_methods = [
            "create_from_documents",
            "load_existing",
            "save",
            "get_retriever",
            "exists",
            "delete",
            "store_type",
        ]

        for method_name in required_methods:
            assert hasattr(VectorStoreInterface, method_name)
            method = getattr(VectorStoreInterface, method_name)
            assert getattr(method, "__isabstractmethod__", False), (
                f"{method_name} should be abstract"
            )


# Integration tests
class TestVectorStoreIntegration:
    """Integration tests for vector store implementations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Product 1: Organic apples from local farm",
                metadata={"product_id": "1", "name": "Organic Apples", "price": 2.99},
            ),
            Document(
                page_content="Product 2: Fresh bananas imported from Ecuador",
                metadata={"product_id": "2", "name": "Fresh Bananas", "price": 1.49},
            ),
        ]

    def test_faiss_end_to_end_workflow(self, temp_dir, sample_documents, monkeypatch):
        """Test complete FAISS workflow from creation to retrieval."""
        # Set environment variables
        monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("FAISS_INDEX_DIR", temp_dir)

        with (
            patch("vector_stores.factory.OpenAIEmbeddings") as mock_embeddings_class,
            patch("vector_stores.faiss_store.FAISS") as mock_faiss_class,
        ):
            # Mock embeddings
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings

            # Mock FAISS
            mock_faiss_instance = Mock()
            mock_faiss_class.from_documents.return_value = mock_faiss_instance
            mock_faiss_class.load_local.return_value = mock_faiss_instance

            # Create store via factory
            store = VectorStoreFactory.create_from_env()

            # Test creation workflow
            store.create_from_documents(sample_documents, mock_embeddings)
            assert store.vector_store == mock_faiss_instance

            # Test save
            store.save()
            mock_faiss_instance.save_local.assert_called_with(temp_dir)

            # Test retrieval
            mock_retriever = Mock()
            mock_faiss_instance.as_retriever.return_value = mock_retriever

            retriever = store.get_retriever(search_type="mmr", k=5)
            assert retriever == mock_retriever

    @patch("vector_stores.pinecone_store.PINECONE_AVAILABLE", True)
    def test_pinecone_end_to_end_workflow(self, sample_documents, monkeypatch):
        """Test complete Pinecone workflow from creation to retrieval."""
        # Set environment variables
        monkeypatch.setenv("VECTOR_STORE_TYPE", "pinecone")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("PINECONE_API_KEY", "test-key")
        monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
        monkeypatch.setenv("PINECONE_REGION", "us-east-1")
        monkeypatch.setenv("PINECONE_CLOUD", "aws")

        with (
            patch("vector_stores.factory.OpenAIEmbeddings") as mock_embeddings_class,
            patch("vector_stores.pinecone_store.Pinecone") as mock_pinecone_class,
            patch(
                "vector_stores.pinecone_store.PineconeVectorStore"
            ) as mock_pinecone_vector_store,
        ):
            # Mock embeddings
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings

            # Mock Pinecone client
            mock_pinecone_instance = Mock()
            mock_pinecone_class.return_value = mock_pinecone_instance
            mock_pinecone_instance.list_indexes.return_value.indexes = []  # Index doesn't exist

            # Mock Pinecone vector store
            mock_vector_store_instance = Mock()
            mock_pinecone_vector_store.from_documents.return_value = (
                mock_vector_store_instance
            )
            mock_pinecone_vector_store.from_existing_index.return_value = (
                mock_vector_store_instance
            )

            # Create store via factory
            store = VectorStoreFactory.create_from_env()

            # Test creation workflow
            with patch.object(store, "_wait_for_index_ready"):
                store.create_from_documents(sample_documents, mock_embeddings)
                assert store.vector_store == mock_vector_store_instance

            # Test save (should do nothing for Pinecone)
            store.save()

            # Test retrieval
            mock_retriever = Mock()
            mock_vector_store_instance.as_retriever.return_value = mock_retriever

            retriever = store.get_retriever(search_type="similarity", k=3)
            assert retriever == mock_retriever

    def test_environment_switching(self, temp_dir, sample_documents, monkeypatch):
        """Test switching between vector store types via environment variables."""
        with patch("vector_stores.factory.OpenAIEmbeddings") as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings

            # Test FAISS
            monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
            monkeypatch.setenv("FAISS_INDEX_DIR", temp_dir)

            with patch("vector_stores.faiss_store.FAISS"):
                faiss_store = VectorStoreFactory.create_from_env()
                assert faiss_store.store_type == "faiss"

            # Test Pinecone
            monkeypatch.setenv("VECTOR_STORE_TYPE", "pinecone")
            monkeypatch.setenv("PINECONE_API_KEY", "test-key")
            monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
            monkeypatch.setenv("PINECONE_REGION", "us-east-1")
            monkeypatch.setenv("PINECONE_CLOUD", "aws")

            with (
                patch("vector_stores.pinecone_store.PINECONE_AVAILABLE", True),
                patch("vector_stores.pinecone_store.Pinecone"),
            ):
                pinecone_store = VectorStoreFactory.create_from_env()
                assert pinecone_store.store_type == "pinecone"
