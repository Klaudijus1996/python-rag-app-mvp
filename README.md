# Python RAG App MVP

A production-ready Retrieval-Augmented Generation (RAG) application built with FastAPI, LangChain, and flexible vector store backends (FAISS/Pinecone) for intelligent grocery and household product recommendations and queries using BigBasket's 28k product dataset.

## 🚀 Features

- **Intelligent Product Recommendations**: AI-powered grocery and household product suggestions based on user queries
- **Multi-Query Support**: Handles recommendations, comparisons, complements, and information queries
- **Session Memory**: Maintains conversation context for follow-up questions
- **FastAPI Backend**: Modern, async API with automatic documentation
- **Flexible Vector Storage**: Support for both FAISS (local) and Pinecone (cloud) vector stores with seamless switching
- **Smart Query Detection**: Automatically detects query types (recommendations, comparisons, complements, information)
- **Vector Search**: Intelligent similarity search for relevant product retrieval
- **Docker Ready**: Separate dev/prod containerized deployment with Docker Compose
- **Structured Logging**: Async logging with daily rotation and JSON support
- **Request Monitoring**: Performance tracking and error monitoring middleware
- **Comprehensive Testing**: Unit and integration tests included
- **LangSmith Integration**: Optional observability and tracing

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │ -> │   FastAPI App   │ -> │   RAG Chain     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                │                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Session Store  │    │ Vector Store    │
                       └─────────────────┘    │   Factory       │
                                             └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ FAISS / Pinecone│
                                              │   + OpenAI      │
                                              │   Embeddings    │
                                              └─────────────────┘
```

## 📋 Prerequisites & Infrastructure

### Required
- **Python 3.11+** - Modern Python with async support
- **OpenAI API key** - For embeddings and chat completions
- **Docker & Docker Compose** - For containerized deployment and development (recommended)

### Vector Store Options
- **FAISS** (default) - Local vector storage, no additional setup required
- **Pinecone** - Cloud vector database requiring API key and index setup

### Optional
- **LangSmith API key** - For observability and tracing
- **Redis** - For distributed session storage (production recommended)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone git@github.com:Klaudijus1996/python-rag-app-mvp.git
cd python-rag-app-mvp
make quickstart
```

This will:

- Install dependencies
- Copy `.env.example` to `.env`
- Run data ingestion
- Start development server

### 2. Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Run data ingestion
python ingest.py

# Start the server
uvicorn app:app --reload --port 8000
```

### 3. Docker Setup

```bash
# Copy environment file and build
make quickstart

# Or manual setup:
cp .env.example .env
# Edit .env with your API keys

# Development environment
make dev

# Production environment  
make prod
```

## 📖 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint with basic information |
| `GET` | `/health` | Health check with system status |
| `POST` | `/chat` | Main RAG query endpoint |
| `POST` | `/ingest` | Data ingestion and vector store creation |
| `GET` | `/retrieve/{query}` | Document retrieval without response generation |
| `GET` | `/sessions/{session_id}` | Get session information |
| `DELETE` | `/sessions/{session_id}` | Clear user session |

### API Usage Examples

#### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "index_ready": true,
  "embedding_model": "text-embedding-3-small",
  "total_chunks": null
}
```

#### Chat with the RAG System

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "query": "I need good storage containers for my kitchen under 200",
    "max_products": 3
  }'
```

#### Advanced Query with Filters

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "query": "recommend cleaning products for household use",
    "category_filter": "Cleaning & Household",
    "price_range": {"min": 100, "max": 300},
    "max_products": 5
  }'
```

#### Retrieve Documents Only

```bash
curl -X GET "http://localhost:8000/retrieve/cleaning%20supplies"
```

#### Session Management

```bash
# Get session info
curl -X GET "http://localhost:8000/sessions/user-123"

# Clear session
curl -X DELETE "http://localhost:8000/sessions/user-123"
```

#### Trigger Re-ingestion

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "force_reindex": true,
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

## 🔧 Configuration

### Environment Variables

#### Core Configuration
| Variable                 | Default                  | Description                     |
| ------------------------ | ------------------------ | ------------------------------- |
| `OPENAI_API_KEY`         | -                        | OpenAI API key (required)       |
| `OPENAI_CHAT_MODEL`      | `gpt-4o-mini`            | Chat model for responses        |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model                 |
| `APP_ENV`                | `dev`                    | Environment (dev/production)    |
| `APP_HOST`               | `0.0.0.0`                | Server host                     |
| `APP_PORT`               | `8000`                   | Server port                     |

#### Vector Store Configuration
| Variable                 | Default                  | Description                     |
| ------------------------ | ------------------------ | ------------------------------- |
| `VECTOR_STORE_TYPE`      | `faiss`                  | Vector store type (faiss/pinecone) |
| `FAISS_INDEX_DIR`        | `store/faiss`            | Local FAISS index directory    |
| `PINECONE_API_KEY`       | -                        | Pinecone API key (if using Pinecone) |
| `PINECONE_ENVIRONMENT`   | -                        | Pinecone environment            |
| `PINECONE_INDEX_NAME`    | `grocery-rag-index`      | Pinecone index name             |

#### RAG Configuration
| Variable                 | Default                  | Description                     |
| ------------------------ | ------------------------ | ------------------------------- |
| `RAG_CHUNK_SIZE`         | `1000`                   | Text chunk size for ingestion   |
| `RAG_CHUNK_OVERLAP`      | `200`                    | Overlap between chunks          |
| `RAG_TOP_K_RESULTS`      | `5`                      | Number of documents to retrieve |
| `RAG_SIMILARITY_THRESHOLD` | `0.7`                  | Minimum similarity threshold    |
| `RAG_MAX_TOKENS_PER_RESPONSE` | `1000`             | Maximum tokens in responses     |

#### Optional Services
| Variable                 | Default                  | Description                     |
| ------------------------ | ------------------------ | ------------------------------- |
| `LANGSMITH_API_KEY`      | -                        | LangSmith API key (optional)    |
| `LANGSMITH_TRACING`      | `true`                   | Enable LangSmith tracing        |
| `LANGSMITH_PROJECT`      | `senukai-rag`            | LangSmith project name          |

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/test_app.py -v
```

## 🐳 Docker Operations

```bash
# Development
make build         # Build development image
make dev           # Start development environment
make dev-with-redis # Start with Redis support

# Production
make build-prod    # Build production image
make prod          # Start production environment
make prod-with-redis # Start production with Redis
make test-prod     # Test production build

# Utilities
make logs          # View development logs
make logs-prod     # View production logs
make shell         # Open shell in dev container
make stop          # Stop all containers
make clean         # Clean up containers and images
```

## 📊 Data Management

### Sample Data Structure

The app expects CSV data with these columns:

```csv
index,product,category,sub_category,brand,sale_price,market_price,type,rating,description
```

### Ingestion Commands

```bash
# Regular ingestion
make ingest

# Force re-ingestion (clears existing index)
make ingest-force
```

## 🎯 Query Types

The system automatically detects and handles different query types:

### 1. Recommendations

```
"I need good hand soap for sensitive skin"
"Recommend storage containers for my pantry"
"Best cleaning wipes under 200"
```

### 2. Comparisons

```
"Compare Nivea vs other soap brands"
"What's the difference between these cleaning wipes?"
"Which storage container is better for food?"
```

### 3. Complements

```
"What cleaning supplies go with these wipes?"
"What storage accessories complement this container?"
"What household items work well with this soap?"
```

### 4. Information

```
"Tell me about this product"
"What are the specifications?"
"How much does it cost?"
```

## 🔍 API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🛠️ Development

### Available Make Commands

#### Quick Start & Setup
```bash
make help           # Show all available commands with descriptions
make setup          # Copy .env.example to .env if needed  
make quickstart     # Complete development setup (build + start)
```

#### Development & Building
```bash
make build          # Build development Docker image
make dev            # Start development environment with validation
make dev-with-redis # Start development with Redis support
make shell          # Open interactive shell in development container
make stop           # Stop all running containers
```

#### Production
```bash
make build-prod     # Build production Docker image
make prod           # Start production environment
make prod-with-redis # Start production with Redis support  
make test-prod      # Test production build with health checks
```

#### Testing & Code Quality
```bash
make test           # Run all tests in development container
make test-coverage  # Run tests with coverage reporting
make lint           # Run code linting with ruff
make lint-fix       # Run linting with automatic fixes
make format         # Format code with ruff
make format-check   # Check code formatting without changes
```

#### Data Management
```bash
make ingest         # Run data ingestion (preserves existing index)
make ingest-force   # Force re-ingestion (clears existing index first)
```

#### Monitoring & Utilities  
```bash
make health         # Check application health endpoint
make logs           # View development container logs (follow mode)
make logs-prod      # View production container logs
make clean          # Clean up containers and images
make clean-all      # Deep clean including volumes and stored data
```

### Code Structure

```
python-rag-app-mvp/
├── app.py                 # FastAPI application with lifespan management and endpoints
├── chains.py              # RAG system with query detection and memory management
├── schema.py              # Pydantic models for API validation and responses
├── ingest.py              # Data ingestion with configurable chunking strategies
├── utils.py               # Data conversion utilities and helper functions
├── logging_config.py      # Structured async logging with daily rotation
├── middleware.py          # Request logging, performance monitoring, error tracking
├── vector_stores/         # Modular vector store implementations
│   ├── __init__.py        # Vector store module exports
│   ├── base.py            # Abstract base interface for vector stores
│   ├── config.py          # Configuration management for vector stores
│   ├── factory.py         # Factory pattern for vector store creation
│   ├── faiss_store.py     # Local FAISS vector store implementation
│   └── pinecone_store.py  # Cloud Pinecone vector store implementation
├── data/
│   ├── big-basket-products-28k.csv  # BigBasket grocery dataset (~28k products)
│   └── products.csv       # Legacy sample data (for backward compatibility)
├── store/
│   └── faiss/             # Local FAISS vector index storage
├── storage/
│   └── logs/              # Application logs with daily rotation and JSON format
├── tests/                 # Comprehensive test suite
│   ├── conftest.py        # Test configuration and fixtures
│   ├── test_app.py        # FastAPI endpoint tests
│   ├── test_chain.py      # RAG system tests
│   ├── test_ingest.py     # Data ingestion tests
│   ├── test_utils.py      # Utility function tests
│   └── test_vector_stores.py  # Vector store implementation tests
├── docker/
│   ├── dev/
│   │   ├── Dockerfile     # Development container configuration
│   │   └── docker-compose.yml  # Dev services with optional Redis
│   └── prod/
│       ├── Dockerfile     # Production optimized container
│       └── docker-compose.yml  # Production services configuration
├── Makefile              # Docker-based development automation
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
└── README.md             # Project documentation
```

### Main Application Components

- **`app.py`**: FastAPI application with async lifespan management, comprehensive middleware stack, and all REST endpoints
- **`chains.py`**: RAG system orchestrating query detection, document retrieval, LLM processing, and session memory
- **`vector_stores/`**: Pluggable vector store architecture supporting both FAISS (local) and Pinecone (cloud) backends
- **`ingest.py`**: Configurable data pipeline for processing CSV data into chunked documents with metadata
- **`middleware.py`**: Request logging, performance monitoring, and error tracking for production observability
- **`logging_config.py`**: Structured async logging with JSON formatting and daily log rotation

## 📈 Performance

### Optimization Tips

1. **Chunk Size**: Adjust `RAG_CHUNK_SIZE` based on your data
2. **Retrieval**: Tune `RAG_TOP_K_RESULTS` for speed vs accuracy
3. **Model Selection**: Use appropriate OpenAI models for your use case
4. **Caching**: Consider Redis for session storage in production

## 🚀 Production Deployment

⚠️ **Production Configuration Status**: The application has basic production Docker setup but requires additional configuration for full production deployment.

### What's Currently Available

✅ **Production Docker Setup**:
```bash
# Build and start production containers
make build-prod
make prod

# Or with Redis support
make prod-with-redis

# Test production deployment
make test-prod
```

✅ **Production Features**:
- Multi-stage Docker build with non-root user
- Health checks configured
- Resource limits set (1GB memory limit)
- Production environment variables
- Log rotation and structured logging

### What Needs Production Configuration

❌ **Security & Networking**:
- CORS origins are currently set to `["*"]` - needs restriction to actual frontend domains
- No reverse proxy (nginx/traefik) configuration provided
- No TLS/SSL certificate management

❌ **Infrastructure**:
- No container orchestration (Kubernetes/Docker Swarm) configuration
- No load balancing setup
- No backup/restore procedures for vector indices

❌ **Monitoring & Logging**:
- No centralized log aggregation setup (ELK/Grafana)
- No metrics collection (Prometheus)
- No alerting configuration

### Basic Production Setup (Docker Compose)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with production API keys

# 2. Update CORS origins in app.py for your domains
# 3. Build and deploy
make build-prod && make prod-with-redis
```

### Recommended Production Architecture

For a complete production deployment, consider:

1. **Container Orchestration**: Kubernetes or Docker Swarm
2. **Load Balancer**: nginx or Traefik with TLS termination  
3. **Vector Storage**: Pinecone for cloud-managed vector storage
4. **Session Storage**: Redis cluster for high availability
5. **Monitoring**: Prometheus + Grafana + AlertManager
6. **Logging**: Centralized logging with ELK stack
7. **Backup**: Regular vector index and data backups

## 🔧 Troubleshooting

### Common Issues

1. **"RAG system not available"**
   - Run `make ingest` to create vector index
   - Check OpenAI API key in `.env`

2. **"No relevant products found"**
   - Verify product data in `data/big-basket-products-28k.csv`
   - Check embedding model configuration

3. **Docker build fails**
   - Ensure Docker has enough memory allocated
   - Check network connectivity for package downloads

4. **Slow responses**
   - Reduce `RAG_TOP_K_RESULTS`
   - Consider using a smaller embedding model

## 🎉 Example Queries

Try these sample queries to explore the system:

```bash
# Product recommendations
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "query": "I need a durable backpack for hiking, around 30L capacity"}'

# Price influencer analysis
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "query": "Why are some coffee grinders more expensive than others?"}'

# Product comparison
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "query": "Compare the features of different smartphone options"}'

# Complementary products
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "query": "What cleaning products would go well with multipurpose wipes?"}'
```
