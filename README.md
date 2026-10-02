# HDFC Mutual Fund FAQ — grounded RAG assistant

A retrieval-augmented assistant that answers questions about five HDFC Mutual Fund
Direct Growth schemes using only facts scraped from a fixed allowlist of Groww
scheme pages. No investment advice, no performance figures, no model-memory answers.

## Status

Phases 0–11 implemented. A FastAPI service (`api.py`) sits beside the Gradio UI so the
assistant can be hosted and called over HTTP.

| Phase | Component |
|---|---|
| 0–2 | Scaffold, source allowlist, fetch, clean, structure extraction |
| 3 | Structure-aware chunker (recursive strategy, 147 chunks) |
| 4 | Pinned MiniLM embedder, 384-dim, 512-token window, disk cache |
| 5 | ChromaDB index + `build_index` orchestrator |
| 6 | Advice, PII, and performance guardrails |
| 7 | Query normalization, synonym expansion, scheme detection |
| 8 | Vector retriever with scheme filtering and score gate |
| 9 | Context builder, strict system prompt, end-to-end chain |
| 10 | Post-generation verification and answer decoration |
| 11 | Gradio chat UI with citations, disclaimer, retrieval trace |
| 12 | `api.py` — REST surface over the same `rag.chain.answer` call |

## LLM provider (open decision O3)

**Chosen: Groq, `qwen/qwen3.8-27b`, over Groq's OpenAI-compatible API.**

- Base URL: `https://api.groq.com/openai/v1`
- Selected for a generous free tier, which keeps the cost ceiling at $0, and for an
  OpenAI-compatible surface, so `rag/llm.py` is a single adapter that can also target
  OpenAI, Together, Ollama, or LM Studio by changing `LLM_PROVIDER`/`LLM_BASE_URL`.
- Decoding is fixed at `temperature=0`, `top_p=1`, single turn, no tools, so answers
  are reproducible for a given context.

### Configuration

Copy `.env.example` to `.env` and set the key:

```
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
LLM_API_KEY=<your key>
```

`LLM_API_KEY` is required for every hosted provider. `rag/llm.py` raises
`LLMNotConfiguredError` rather than silently degrading when the key is empty.
Ollama and LM Studio run without a key.

## Running

Use **Python 3.11 or 3.12** — that is what the container runs. On 3.13/3.14,
`chromadb` 1.0.20 still uses the `pydantic.v1` shim, which breaks on the
`pydantic<=2.12.3` that `gradio` 5.50.0 requires. If you must stay on 3.14, install
`gradio==5.50.0` with `pydantic==2.13.5` and accept the declared-cap conflict; the
app runs, but `pip install -r requirements-dev.txt` will undo it.

`requirements.txt` is the runtime set the container installs — no UI, no test runner.
`requirements-dev.txt` adds `gradio` for `app.py` and `pytest` for the suite. Install
the dev set to work on this repo; the API and its tests run on the runtime set alone.

```bash
pip install -r requirements-dev.txt

python -m ingest.build_index          # fetch, chunk, embed, persist, verify
python -m ingest.build_index --verify # verify the existing index only

python -m rag.retriever --probe "what is the exit load on HDFC large cap"
python -c "from rag.chain import answer; print(answer('what is the expense ratio of HDFC Large Cap Fund?').text)"

python -m pytest tests/ -v
```

## HTTP API

`api.py` exposes the same `rag.chain.answer` path the Gradio UI uses, so guardrails,
the score gate, and citation decoration behave identically on both surfaces.

```bash
python -m api                                    # http://127.0.0.1:8000/docs
```

| Route | Purpose |
|---|---|
| `GET /health` | Liveness plus index state, source count, provider, model, corpus date |
| `GET /sources` | The five allowlisted scheme pages |
| `POST /ask` | `{"question": "...", "include_trace": true}` → answer, citations, trace |

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the expense ratio of HDFC Large Cap Fund - Direct Growth?","include_trace":false}'
```

`/ask` returns `200` with `refusal` and `refusal_type` set for guardrail and
out-of-corpus answers, `422` for a blank or oversized question, `503` when the index
or the LLM is unavailable, and `500` otherwise. A PII refusal never echoes the
submitted question back — the `question` field is replaced with a redaction marker.

| Variable | Default | Notes |
|---|---|---|
| `PORT` | `8000` | Render and Railway inject this |
| `API_HOST` | `0.0.0.0` | |
| `API_WORKERS` | `1` | One worker keeps the Chroma client and model single-resident |
| `API_WARMUP` | `1` | Loads the embedder at startup so request one is not slow |
| `CORS_ORIGINS` | `*` | Comma-separated allowlist for a hosted frontend |
| `CHROMA_DIR` | `./chroma_db` | Point at a mounted volume in production |
| `ARTIFACTS_DIR` | `./artifacts` | Holds `corpus_meta.json` |
| `SOURCES_CSV` | `./ingest/sources.csv` | |

## Deploying

`Dockerfile` installs the CPU-only torch wheel, bakes the MiniLM model into the image
at `HF_HOME=/opt/hf`, builds the Chroma index during the image build, and starts with
a verify-then-rebuild guard so a stale or missing index self-heals on boot. The
container listens on `$PORT`. It copies `config.py`, `api.py`, `ingest/`, and `rag/`
only — `app.py` and the Gradio dependency stay out of the API image, which is worth
about 430 MB.

```bash
docker build -t hdfc-mf-faq .
docker run --rm -p 8000:8000 --env-file .env hdfc-mf-faq
```

`render.yaml` is a Render blueprint for the same image. On Render use the `standard`
plan: the free plan's 512 MB cannot hold torch plus the embedder.

```bash
git push                                   # Render builds from the repo
# then in Render: New > Blueprint > select this repo > Apply
```

Railway works from the same Dockerfile — set the root directory to the repo root,
build with the Dockerfile, and mount a volume at `/app/chroma_db` if you want the
index to survive restarts.

## Architecture notes

- **Layering.** `ingest/` builds the index; `rag/` answers questions. `rag/retriever.py`
  imports `ingest.embedder` so the query model is guaranteed identical to the indexed
  one, and asserts `embed_model`, `embed_dim`, and `embed_max_seq_length` match at
  startup. `ingest/` importing shared `rag/schemas.py` and `rag/logging_utils.py` is a
  known deviation from the documented one-way dependency rule.
- **Never trust the collection's embedder.** Chroma silently attaches its own ONNX
  embedding function. Every collection is opened with `embedding_function=None` and all
  queries pass explicit vectors from `ingest.embedder`.
- **Refusals never call the LLM.** Advice, PII, performance, and out-of-corpus answers
  are fixed strings, so they are byte-identical every time. `Answer.trace.guards`
  records `llm_called` for every answer, and tests assert it is `False` on every
  refusal path.
- **PII is never logged or returned.** `detect_pii` returns pattern names only, and a
  PII hit discards the query so it is never embedded or stored.
- **Retrieval is not bit-reproducible.** Chroma's HNSW index returns different chunks
  at tied distances across processes. Assert on distances, not on exact top-k
  membership.
- **First call is slow.** The embedding model loads lazily (~15 s); warm queries run
  in roughly 50 ms.

## Safety

Facts-only. No investment advice.
