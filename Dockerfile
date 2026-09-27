FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    PORT=8000 \
    API_HOST=0.0.0.0 \
    ARTIFACTS_DIR=/app/artifacts \
    CHROMA_DIR=/app/chroma_db

WORKDIR /app

RUN pip install --no-cache-dir torch==2.9.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py app.py api.py ./
COPY ingest/ ingest/
COPY rag/ rag/

RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

RUN python -m ingest.build_index || echo "index build deferred to first container start"

EXPOSE 8000

CMD ["sh", "-c", "python -m ingest.build_index --verify || python -m ingest.build_index || true; exec python -m api"]
