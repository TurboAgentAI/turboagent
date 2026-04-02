# =============================================================================
# TurboAgent API Server — Docker image
#
# Build:
#   docker build -t turboagent .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -e TURBOAGENT_MODEL=Qwen/Qwen2.5-32B-Instruct \
#     turboagent
#
# With auth:
#   docker run --gpus all -p 8000:8000 \
#     -e TURBOAGENT_MODEL=Qwen/Qwen2.5-32B-Instruct \
#     -e TURBOAGENT_API_KEYS=sk-key1,sk-key2 \
#     turboagent
# =============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install TurboAgent with server + torch + native extras
COPY pyproject.toml setup.py README.md LICENSE ./
COPY turboagent/ turboagent/

RUN pip install --no-cache-dir -e ".[server,torch,native]"

# HuggingFace cache (mount a volume for persistence)
ENV HF_HOME=/data/hf_cache
RUN mkdir -p /data/hf_cache

# Default config (override via -e or docker-compose)
ENV TURBOAGENT_MODEL=""
ENV TURBOAGENT_BACKEND=""
ENV TURBOAGENT_KV_MODE="turbo3"
ENV TURBOAGENT_CONTEXT=""
ENV TURBOAGENT_API_KEYS=""
ENV TURBOAGENT_RATE_LIMIT="60"

EXPOSE 8000

COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
