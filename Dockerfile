# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 9995

# Set environment variables with defaults
ENV QDRANT_URL=http://localhost:6333
ENV QDRANT_COLLECTION=knowledge_base
ENV RAG_TOP_K=5
ENV RAG_CONTEXT_CHAR_LIMIT=8000

# Run the application
CMD ["python", "main.py"]

