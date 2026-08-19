# Slim rather than alpine: the ML wheels are built against glibc, and on alpine
# pip falls back to compiling from source, which turns a two-minute build into
# twenty.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Requirements first, so the dependency layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# Bake the embedding model into the image. Without this the first request after
# every container start pays a model download, which makes the p95 latency in the
# README a fiction.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# The corpus is a mounted volume, not an image layer -- it is not ours to
# redistribute inside an image.
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"

CMD ["uvicorn", "rageval.api:app", "--host", "0.0.0.0", "--port", "8000"]
