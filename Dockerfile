# Quiz Generator API container.
#
# We deliberately do NOT bake the Chroma DB, processed JSONL, or downloaded
# model files into the image — they're large, change often, and would make
# every code rebuild slow. They're attached as volumes via docker-compose.
#
# Image size after a clean build is ~3 GB, dominated by torch +
# sentence-transformers (which we need for BGE-M3 + reranker).

FROM python:3.11-slim AS runtime

# Some Python ML libraries (torch, faiss) need libgomp at runtime for
# multi-threaded numerics. Install it once, throw away the apt cache.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies FIRST, separately from the source code.
# Docker caches this layer, so editing src/*.py doesn't reinstall torch.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy only what's needed at runtime — keep the image lean.
COPY src/ ./src/
COPY configs/ ./configs/

# Sentence-transformers caches downloaded models under HF_HOME.
# We point it at /cache/huggingface, which docker-compose mounts as a
# named volume so models persist across container restarts (the alternative
# is re-downloading 1.2 GB of BGE-M3 + reranker every time).
ENV HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Bind to 0.0.0.0 so other containers on the docker network can reach us.
# Inside the container, that's safe — it's only exposed to the host via
# the port mapping in docker-compose.yml.
CMD ["python", "-m", "src.api", "--host", "0.0.0.0", "--port", "8000"]
