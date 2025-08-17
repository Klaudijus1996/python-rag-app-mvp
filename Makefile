# Makefile for Python RAG App MVP

.PHONY: help install dev test clean build run ingest docker-build docker-run docker-stop lint format

# Default target
help:
	@echo "Python RAG App MVP - Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     - Install dependencies (local)"
	@echo "  dev         - Run in development mode (local)"
	@echo "  test        - Run tests (in Docker container)"
	@echo "  lint        - Run linting (in Docker container)"
	@echo "  format      - Format code (in Docker container)"
	@echo ""
	@echo "Data:"
	@echo "  ingest      - Run data ingestion (in Docker container)"
	@echo "  ingest-force - Force re-ingestion (in Docker container)"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  docker-stop  - Stop Docker containers"
	@echo "  docker-logs  - View Docker logs"
	@echo ""
	@echo "Production:"
	@echo "  build       - Build production image"
	@echo "  run         - Run production server"
	@echo "  clean       - Clean generated files"

# Development setup
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

dev:
	@echo "Starting development server..."
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Testing
test:
	@echo "Running tests in Docker container..."
	@docker-compose ps | grep -q "rag-app.*Up" || { echo "Container not running. Starting..."; docker-compose up -d; sleep 5; }
	docker-compose exec rag-app python -m pytest -v tests/

test-coverage:
	@echo "Running tests with coverage in Docker container..."
	@docker-compose ps | grep -q "rag-app.*Up" || { echo "Container not running. Starting..."; docker-compose up -d; sleep 5; }
	docker-compose exec rag-app python -m pytest --cov=. --cov-report=html --cov-report=term tests/

# Code quality
lint:
	@echo "Running linting in Docker container..."
	docker-compose exec rag-app sh -c "if command -v ruff >/dev/null 2>&1; then ruff check .; elif command -v flake8 >/dev/null 2>&1; then flake8 .; else echo 'No linter found. Install ruff or flake8'; fi"

format:
	@echo "Formatting code in Docker container..."
	docker-compose exec rag-app sh -c "if command -v ruff >/dev/null 2>&1; then ruff format .; elif command -v black >/dev/null 2>&1; then black .; else echo 'No formatter found. Install ruff or black'; fi"

# Data ingestion
ingest:
	@echo "Running data ingestion in Docker container..."
	docker-compose exec rag-app python ingest.py

ingest-force:
	@echo "Force re-ingesting data in Docker container..."
	docker-compose exec rag-app rm -rf store/faiss
	docker-compose exec rag-app python ingest.py

# Docker operations
docker-build:
	@echo "Building Docker image..."
	docker build -t python-rag-app-mvp .

docker-run:
	@echo "Starting with Docker Compose..."
	docker-compose up -d

docker-run-with-redis:
	@echo "Starting with Docker Compose (including Redis)..."
	docker-compose --profile with-redis up -d

docker-stop:
	@echo "Stopping Docker containers..."
	docker-compose down

docker-logs:
	@echo "Viewing Docker logs..."
	docker-compose logs -f

docker-shell:
	@echo "Opening shell in container..."
	docker-compose exec rag-app bash

# Production
build: docker-build

run:
	@echo "Starting production server..."
	python app.py

# Cleanup
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

clean-all: clean
	@echo "Deep cleaning..."
	rm -rf store/
	rm -rf storage/logs/*

# Environment setup
setup-env:
	@echo "Setting up environment..."
	cp .env.example .env
	@echo "Please edit .env file with your API keys"

# Quick start
quickstart: install setup-env ingest dev

# Health check
health:
	@echo "Checking application health..."
	@curl -f http://localhost:8000/health || echo "Application is not running"

# View logs
logs:
	@echo "Viewing application logs..."
	@if [ -f "storage/logs/app.log" ]; then \
		tail -f storage/logs/app.log; \
	else \
		echo "No log file found"; \
	fi

# Dependencies update
update-deps:
	@echo "Updating dependencies..."
	pip-compile requirements.in --upgrade || pip freeze > requirements.txt

# Database/Index management
reset-index:
	@echo "Resetting vector index..."
	rm -rf store/faiss
	$(MAKE) ingest

# Performance testing
perf-test:
	@echo "Running performance tests..."
	@if command -v ab >/dev/null 2>&1; then \
		ab -n 100 -c 10 http://localhost:8000/health; \
	else \
		echo "Apache Bench (ab) not found. Install apache2-utils"; \
	fi

# Security scan
security-scan:
	@echo "Running security scan..."
	@if command -v safety >/dev/null 2>&1; then \
		safety check; \
	else \
		echo "Safety not found. Install with: pip install safety"; \
	fi

# Documentation
docs:
	@echo "Starting documentation server..."
	@echo "API docs available at: http://localhost:8000/docs"
	@echo "ReDoc available at: http://localhost:8000/redoc"