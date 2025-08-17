# Python RAG App MVP

A production-ready Retrieval-Augmented Generation (RAG) application built with FastAPI, LangChain, and FAISS for intelligent product recommendations and queries.

## 🚀 Features

- **Intelligent Product Recommendations**: AI-powered product suggestions based on user queries
- **Price Influencer Analysis**: Explains pricing factors like brand, materials, features, and capacity
- **Multi-Query Support**: Handles recommendations, comparisons, complements, and information queries
- **Session Memory**: Maintains conversation context for follow-up questions
- **FastAPI Backend**: Modern, async API with automatic documentation
- **Vector Search**: FAISS-powered similarity search for relevant product retrieval
- **Docker Ready**: Containerized deployment with Docker Compose
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
# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Build and run
make docker-build
make docker-run

# Or use docker-compose directly
docker-compose up -d
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
    "query": "I need a durable 1L stainless steel water bottle under 30 EUR",
    "max_products": 3
  }'
```

### Advanced Query with Filters

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123",
    "query": "recommend a laptop for programming",
    "category_filter": "Electronics",
    "price_range": {"min": 500, "max": 1500},
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
# Build image
make docker-build

# Run with Docker Compose
make docker-run

# Run with Redis (production)
make docker-run-with-redis

# View logs
make docker-logs

# Stop containers
make docker-stop

# Open shell in container
make docker-shell
```

## 📊 Data Management

### Sample Data Structure

The app expects CSV data with these columns:

```csv
product_id,name,category,brand,price,currency,description,features,materials,capacity,compatibility,variants,stock,sku,url,image_url,tags
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
"I need a laptop for gaming"
"Recommend a water bottle for hiking"
"Best smartphone under $500"
```

### 2. Comparisons

```
"Compare iPhone vs Samsung Galaxy"
"What's the difference between these laptops?"
"Which is better: Product A or Product B?"
```

### 3. Complements

```
"What accessories go with this laptop?"
"Compatible products for this camera"
"What pairs well with this coffee maker?"
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
make dev            # Start development server
make test           # Run tests
make lint           # Run code linting
make format         # Format code
make clean          # Clean generated files
make health         # Check application health
```

### Code Structure

```
python-rag-app-mvp/
├── app.py                 # FastAPI application
├── chains.py              # RAG pipeline and prompts
├── schema.py              # Pydantic models
├── ingest.py              # Data ingestion script
├── data/products.csv      # Sample product catalog
├── store/                 # FAISS vector index
├── tests/                 # Test suite
├── storage/logs/          # Application logs
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container setup
├── Makefile              # Development automation
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

   - Verify product data in `data/products.csv`
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

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🤝 Support

For issues and questions:

1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information

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
  -d '{"session_id": "demo", "query": "What accessories would go well with a laptop for remote work?"}'
```
