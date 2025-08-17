# Makefile for Python RAG App MVP
# All commands run in Docker containers - no local Python dependencies needed

# Docker configuration
DOCKER_COMPOSE = docker compose
DOCKER_DIR = docker
DEV_COMPOSE_FILE = $(DOCKER_DIR)/docker-compose.dev.yml
PROD_COMPOSE_FILE = $(DOCKER_DIR)/docker-compose.prod.yml

.PHONY: help setup quickstart build build-prod dev prod test test-coverage lint format ingest clean stop logs shell health

# Default target
help:
	@echo "Python RAG App MVP - Available commands:"
	@echo ""
	@echo "Quick Start:"
	@echo "  setup       - Copy .env.example to .env if needed"
	@echo "  quickstart  - Full development setup and start"
	@echo ""
	@echo "Development:"
	@echo "  build       - Build development Docker image"
	@echo "  dev         - Start development environment"
	@echo "  test        - Run tests in development container"
	@echo "  test-coverage - Run tests with coverage"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  ingest      - Run data ingestion"
	@echo ""
	@echo "Production:"
	@echo "  build-prod  - Build production Docker image"
	@echo "  prod        - Start production environment"
	@echo "  test-prod   - Test production build"
	@echo ""
	@echo "Utilities:"
	@echo "  stop        - Stop all containers"
	@echo "  logs        - View container logs"
	@echo "  shell       - Open shell in development container"
	@echo "  health      - Check application health"
	@echo "  clean       - Clean up containers and images"

# Environment setup
setup:
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then \
		if [ -f .env.example ]; then \
			cp .env.example .env; \
			echo "✓ Created .env from .env.example"; \
			echo "⚠️  Please edit .env file with your API keys"; \
		else \
			echo "❌ .env.example not found. Please create .env manually"; \
			exit 1; \
		fi; \
	else \
		echo "✓ .env file already exists"; \
	fi

# Validate required environment variables
validate-env:
	@echo "Validating environment variables..."
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found. Run 'make setup' first"; \
		exit 1; \
	fi
	@if ! grep -q "OPENAI_API_KEY=" .env || grep -q "OPENAI_API_KEY=$$" .env; then \
		echo "⚠️  OPENAI_API_KEY not set in .env file"; \
		echo "Please add your OpenAI API key to continue"; \
		exit 1; \
	fi
	@echo "✓ Environment variables validated"

# Development commands
build:
	@echo "Building development Docker image..."
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) build

dev: setup validate-env
	@echo "Starting development environment..."
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) up -d
	@echo "✓ Development server started at http://localhost:8000"
	@echo "  API docs: http://localhost:8000/docs"
	@echo "  ReDoc: http://localhost:8000/redoc"

# Production commands
build-prod:
	@echo "Building production Docker image..."
	$(DOCKER_COMPOSE) -f $(PROD_COMPOSE_FILE) build

prod: setup validate-env
	@echo "Starting production environment..."
	$(DOCKER_COMPOSE) -f $(PROD_COMPOSE_FILE) up -d
	@echo "✓ Production server started at http://localhost:8000"

test-prod: build-prod
	@echo "Testing production build..."
	$(DOCKER_COMPOSE) -f $(PROD_COMPOSE_FILE) up -d
	@sleep 10
	@echo "Running health check..."
	@if curl -f http://localhost:8000/health > /dev/null 2>&1; then \
		echo "✓ Production build test passed"; \
	else \
		echo "❌ Production build test failed"; \
		exit 1; \
	fi
	@$(DOCKER_COMPOSE) -f $(PROD_COMPOSE_FILE) down

# Testing
test:
	@echo "Running tests in development container..."
	@$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) ps | grep -q "rag-app.*Up" || { echo "Starting dev container..."; $(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) up -d; sleep 5; }
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app python -m pytest -v tests/

test-coverage:
	@echo "Running tests with coverage..."
	@$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) ps | grep -q "rag-app.*Up" || { echo "Starting dev container..."; $(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) up -d; sleep 5; }
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app python -m pytest --cov=. --cov-report=html --cov-report=term tests/

# Code quality
lint:
	@echo "Running linting..."
	@$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) ps | grep -q "rag-app.*Up" || { echo "Starting dev container..."; $(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) up -d; sleep 5; }
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app ruff check .

lint-fix:
	@echo "Running linting with auto-fix..."
	@$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) ps | grep -q "rag-app.*Up" || { echo "Starting dev container..."; $(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) up -d; sleep 5; }
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app ruff check . --fix

format:
	@echo "Formatting code..."
	@$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) ps | grep -q "rag-app.*Up" || { echo "Starting dev container..."; $(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) up -d; sleep 5; }
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app ruff format .

format-check:
	@echo "Checking code formatting..."
	@$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) ps | grep -q "rag-app.*Up" || { echo "Starting dev container..."; $(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) up -d; sleep 5; }
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app ruff format . --check

# Data ingestion
ingest:
	@echo "Running data ingestion..."
	@$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) ps | grep -q "rag-app.*Up" || { echo "Starting dev container..."; $(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) up -d; sleep 5; }
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app python ingest.py

ingest-force:
	@echo "Force re-ingesting data..."
	@$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) ps | grep -q "rag-app.*Up" || { echo "Starting dev container..."; $(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) up -d; sleep 5; }
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app rm -rf store/faiss
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app python ingest.py

# Quick start - complete development setup
quickstart: setup validate-env build
	@echo "Starting quickstart setup..."
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) up -d
	@echo "Waiting for container to be ready..."
	@sleep 10
	@echo "Running data ingestion..."
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app python ingest.py
	@echo ""
	@echo "🚀 Quickstart complete!"
	@echo "   Development server: http://localhost:8000"
	@echo "   API docs: http://localhost:8000/docs"
	@echo "   ReDoc: http://localhost:8000/redoc"

# Utility commands
stop:
	@echo "Stopping all containers..."
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) down
	$(DOCKER_COMPOSE) -f $(PROD_COMPOSE_FILE) down

logs:
	@echo "Viewing development container logs..."
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) logs -f

logs-prod:
	@echo "Viewing production container logs..."
	$(DOCKER_COMPOSE) -f $(PROD_COMPOSE_FILE) logs -f

shell:
	@echo "Opening shell in development container..."
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) exec rag-app bash

health:
	@echo "Checking application health..."
	@if curl -f http://localhost:8000/health > /dev/null 2>&1; then \
		echo "✓ Application is healthy"; \
	else \
		echo "❌ Application is not responding"; \
		exit 1; \
	fi

# Cleanup
clean:
	@echo "Cleaning up containers and images..."
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) down -v
	$(DOCKER_COMPOSE) -f $(PROD_COMPOSE_FILE) down -v
	docker system prune -f
	@echo "✓ Cleanup complete"

clean-all: clean
	@echo "Deep cleaning..."
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) down -v --rmi all
	$(DOCKER_COMPOSE) -f $(PROD_COMPOSE_FILE) down -v --rmi all
	rm -rf store/*
	rm -rf storage/logs/*

# Redis support
dev-with-redis: setup validate-env
	@echo "Starting development environment with Redis..."
	$(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) --profile with-redis up -d

prod-with-redis: setup validate-env
	@echo "Starting production environment with Redis..."
	$(DOCKER_COMPOSE) -f $(PROD_COMPOSE_FILE) --profile with-redis up -d