# ==============================================================================
# Togyzkumalak Engine - Docker Image
# ==============================================================================
# Base: PyTorch with CUDA support for GPU-accelerated training
# Use with Vast.ai PyTorch template or any NVIDIA GPU server
# ==============================================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models/alphazero \
    /app/logs/alphazero \
    /app/logs/games \
    /app/logs/gemini_battles/games \
    /app/logs/gemini_battles/sessions \
    /app/logs/gemini_battles/summaries \
    /app/logs/self_play \
    /app/logs/training \
    /app/logs/wandb_local \
    /app/logs/ab_tests/results \
    /app/training_data

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command - run the FastAPI server
CMD ["python", "run.py"]
