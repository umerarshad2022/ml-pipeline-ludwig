FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pipelines/ ./pipelines/
COPY examples/ ./examples/
COPY run_pipeline.py .

# Create directories
RUN mkdir -p artifacts logs data results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LUDWIG_CACHE_DIR=/app/.ludwig_cache

# Default command
CMD ["python", "run_pipeline.py", "examples/du-cell-ludwig.yaml"]
