# Development Dockerfile for Python RAG App MVP
FROM python:3.11-slim

# Install system dependencies including build tools for development
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional development dependencies
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    debugpy

# Copy application code (will be overridden by volume mount in dev)
COPY . .

# Create necessary directories
RUN mkdir -p store tests storage/logs

# Expose port
EXPOSE 8000

# Development command (can be overridden)
CMD ["python", "app.py"]