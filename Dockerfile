FROM python:3.12-slim-bookworm

# Install system dependencies for tree-sitter and git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
        ca-certificates && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy workspace definition and lockfile first for better Docker layer caching
COPY pyproject.toml uv.lock ./

# Copy workspace members
COPY fastcode/ fastcode/
COPY nanobot/ nanobot/

# Install dependencies (no dev, frozen lockfile)
RUN uv sync --frozen --no-dev

# Pre-download the embedding model BEFORE copying app code
# so that code changes won't invalidate this ~470MB cached layer
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Create necessary directories
RUN mkdir -p /app/repos /app/data /app/logs

# Copy configuration
COPY config/ config/

# Default port for FastCode API
EXPOSE 8001

# Environment defaults (can be overridden in docker-compose)
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

CMD ["uv", "run", "fastcode-api", "--host", "0.0.0.0", "--port", "8001"]
