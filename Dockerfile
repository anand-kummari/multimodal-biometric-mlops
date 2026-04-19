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
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification first (layer caching)
COPY pyproject.toml README.md ./
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime Stage ----
FROM python:3.12-slim AS runtime

LABEL maintainer="Anand Kummari" \
      description="Multimodal biometric recognition MLOps pipeline"

WORKDIR /app

# Install tini for proper PID-1 signal handling and minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser \
    && mkdir -p data/raw data/processed data/cache checkpoints \
    && chown -R appuser:appuser /app
USER appuser

# Set Python path
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Basic health check — verify the package can be imported
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import biometric" || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "biometric", "--help"]
