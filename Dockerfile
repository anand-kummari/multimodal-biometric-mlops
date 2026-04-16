# Multi-stage Docker build for the multimodal biometric MLOps pipeline
#
# Stage 1 (builder): Installs all dependencies and builds the package
# Stage 2 (runtime): Minimal image with only runtime dependencies
#
# Usage:
#   docker build -t multimodal-biometric-mlops:latest .
#   docker run --gpus all -v ./data:/app/data multimodal-biometric-mlops:latest \
#       python scripts/train.py training=quick

# ---- Builder Stage ----
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification first (layer caching)
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime Stage ----
FROM python:3.10-slim as runtime

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Create data directories
RUN mkdir -p data/raw data/processed data/cache checkpoints

# Set Python path
ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

# Default command: show help
CMD ["python", "-m", "biometric", "--help"]
