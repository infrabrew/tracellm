# TraceLLM — Multi-stage Docker build
# Supports both CPU and GPU (CUDA) builds.
#
# Build:
#   docker build -t tracellm -f docker/Dockerfile .
#   docker build -t tracellm:gpu --build-arg BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 -f docker/Dockerfile .
#
# Run:
#   docker run -p 8400:8400 tracellm
#   docker run --gpus all -p 8400:8400 tracellm:gpu

ARG BASE=python:3.11-slim

# ── Stage 1: Build ───────────────────────────────────────────────────────────
FROM ${BASE} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy config and prompts
WORKDIR /app
COPY config/ config/
COPY prompts/ prompts/

# Create model cache directory
RUN mkdir -p /root/.tracellm/models /root/.tracellm/checkpoints /root/.tracellm/logs

# Expose server port
EXPOSE 8400

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8400/health')" || exit 1

ENTRYPOINT ["tracellm"]
CMD ["serve"]
