#!/bin/bash

# Docker entrypoint script for RAG application

set -e

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if vector index exists
check_index() {
    if [ -d "/app/store/faiss" ] && [ "$(ls -A /app/store/faiss 2>/dev/null)" ]; then
        return 0
    else
        return 1
    fi
}

# Function to run ingestion
run_ingestion() {
    log "Running data ingestion..."
    python ingest.py
    if [ $? -eq 0 ]; then
        log "Ingestion completed successfully"
    else
        log "Ingestion failed"
        exit 1
    fi
}

# Function to start the application
start_app() {
    log "Starting RAG application..."
    exec python app.py
}

# Main execution
log "Starting RAG application container..."

# Check if we should run ingestion
if [ "$AUTO_INGEST" = "true" ] || [ "$1" = "ingest" ]; then
    log "Auto-ingestion enabled or requested"
    if ! check_index; then
        log "Vector index not found, running ingestion..."
        run_ingestion
    else
        log "Vector index already exists, skipping ingestion"
    fi
elif [ "$1" = "ingest-force" ]; then
    log "Forced ingestion requested"
    run_ingestion
fi

# Handle different command options
case "$1" in
    "ingest")
        log "Running ingestion only"
        run_ingestion
        ;;
    "ingest-force")
        log "Running forced ingestion only"
        run_ingestion
        ;;
    "app" | "")
        # Check if index exists before starting app
        if ! check_index; then
            log "WARNING: Vector index not found. Application will start but chat endpoint may not work."
            log "Run 'docker-compose exec rag-app python ingest.py' to create the index."
        fi
        start_app
        ;;
    "test")
        log "Running tests..."
        exec pytest -v
        ;;
    "shell")
        log "Starting interactive shell..."
        exec /bin/bash
        ;;
    *)
        log "Running custom command: $*"
        exec "$@"
        ;;
esac