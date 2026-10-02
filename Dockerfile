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

COPY config.py api.py ./
COPY ingest/ ingest/
COPY rag/ rag/

# chromadb eagerly loads a default ONNX embedding function and a Kubernetes client
# that this project never uses: every collection is opened with
# embedding_function=None and persisted to local disk. Dropping both removes about
# 150 MB of image and, because onnxruntime is loaded at import time, roughly
# 240 MB of resident memory. The import check fails the build loudly if a future
# chromadb release makes either one mandatory.
RUN pip uninstall -y kubernetes onnxruntime \
 && python -c "import chromadb, api, ingest.embedder, rag.chain"

RUN ok=0; \
    for i in 1 2 3 4 5; do \
      if python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"; then \
        ok=1; break; \
      fi; \
      echo "model_download_attempt_${i}_failed"; sleep $((i * 10)); \
    done; \
    if [ "$ok" != "1" ]; then echo "model_bake_failed"; exit 1; fi

RUN python -m ingest.build_index || echo "index build deferred to first container start"

EXPOSE 8000

CMD ["sh", "-c", "python -m ingest.build_index --verify || python -m ingest.build_index || true; exec python -m api"]
