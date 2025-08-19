# Python RAG App MVP

A production-ready Retrieval-Augmented Generation (RAG) application built with FastAPI, LangChain, and FAISS for intelligent grocery and household product recommendations and queries.

## 🚀 Features

- **Intelligent Product Recommendations**: AI-powered grocery and household product suggestions based on user queries
- **Price Influencer Analysis**: Explains pricing factors like brand reputation, nutritional value, pack size, and organic/premium quality
- **Multi-Query Support**: Handles recommendations, comparisons, complements, and information queries
- **Session Memory**: Maintains conversation context for follow-up questions
- **FastAPI Backend**: Modern, async API with automatic documentation
- **Vector Search**: FAISS-powered similarity search for relevant product retrieval
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
                       │  Session Store  │    │ FAISS Retriever │
                       └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ OpenAI Embeddings│
                                              └─────────────────┘
```

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key
- (Optional) LangSmith API key for observability

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

## 📖 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Chat with the RAG System

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "query": "I need good storage containers for my kitchen under 200",
    "max_products": 3
  }'
```

### Advanced Query with Filters

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

### Trigger Re-ingestion

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

| Variable                 | Default                  | Description                     |
| ------------------------ | ------------------------ | ------------------------------- |
| `OPENAI_API_KEY`         | -                        | OpenAI API key (required)       |
| `OPENAI_CHAT_MODEL`      | `gpt-4o-mini`            | Chat model for responses        |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model                 |
| `LANGSMITH_API_KEY`      | -                        | LangSmith API key (optional)    |
| `LANGSMITH_TRACING`      | `true`                   | Enable LangSmith tracing        |
| `LANGSMITH_PROJECT`      | `senukai-rag`            | LangSmith project name          |
| `RAG_CHUNK_SIZE`         | `1000`                   | Text chunk size for ingestion   |
| `RAG_CHUNK_OVERLAP`      | `200`                    | Overlap between chunks          |
| `RAG_TOP_K_RESULTS`      | `5`                      | Number of documents to retrieve |
| `APP_HOST`               | `0.0.0.0`                | Server host                     |
| `APP_PORT`               | `8000`                   | Server port                     |

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

# Reset index completely
make reset-index
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

```bash
make help           # Show all available commands
make quickstart     # Complete development setup
make dev            # Start development environment
make test           # Run tests in container
make test-coverage  # Run tests with coverage
make lint           # Run code linting
make lint-fix       # Run linting with auto-fix
make format         # Format code
make format-check   # Check code formatting
make ingest         # Run data ingestion
make ingest-force   # Force re-ingestion
make health         # Check application health
make clean          # Clean containers and images
```

### Code Structure

```
python-rag-app-mvp/
├── app.py                 # FastAPI application with lifespan management
├── chains.py              # RAG pipeline and prompts
├── schema.py              # Pydantic models
├── ingest.py              # Data ingestion script
├── logging_config.py      # Structured logging with async support
├── middleware.py          # Request logging and monitoring middleware
├── data/big-basket-products-28k.csv      # Sample product catalog consting of almost 28k products
├── store/faiss/           # FAISS vector index
├── storage/logs/          # Application logs with daily rotation
├── tests/                 # Test suite
├── docker/
│   ├── dev/               # Development Docker configuration
│   └── prod/              # Production Docker configuration
├── Makefile              # Docker-based development automation
└── README.md             # This file
```

## 📈 Performance

### Optimization Tips

1. **Chunk Size**: Adjust `RAG_CHUNK_SIZE` based on your data
2. **Retrieval**: Tune `RAG_TOP_K_RESULTS` for speed vs accuracy
3. **Model Selection**: Use appropriate OpenAI models for your use case
4. **Caching**: Consider Redis for session storage in production

### Benchmarking

```bash
# Basic performance test
make perf-test

# Custom load test
ab -n 1000 -c 10 http://localhost:8000/health
```

## 🔒 Security

```bash
# Run security scan
make security-scan

# Check for vulnerabilities
safety check
```

## 🚀 Production Deployment

### Docker Compose (Recommended)

```yaml
# Use the included docker-compose.yml
docker-compose --profile with-redis up -d
```

### Manual Deployment

1. Set `APP_ENV=production`
2. Configure proper CORS origins
3. Use a reverse proxy (nginx/traefik)
4. Set up persistent storage for vector index
5. Configure log aggregation
6. Enable health checks and monitoring

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

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python app.py

# View detailed logs
make logs
```

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
