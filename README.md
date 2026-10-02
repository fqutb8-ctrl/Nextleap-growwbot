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
container listens on `$PORT`.

`api.py` mounts the Gradio chat UI from `app.py` at `/` via
`gr.mount_gradio_app`, so one container serves both surfaces: the UI at `/`, Swagger at
`/docs`, and the JSON API at `/ask`, `/health`, and `/sources`. The mount happens at
import time, so the UI is present however `api:app` is launched. Because `/` is a
catch-all mount, every API route must stay declared above it.

```bash
docker build -t hdfc-mf-faq .
docker run --rm -p 8000:8000 --env-file .env hdfc-mf-faq
```

`render.yaml` is a Render blueprint for the same image, on the `free` plan. Measured
resident memory against the 512 MB free instance, with the Gradio UI mounted at `/`:

| State | Memory | Share |
|---|---|---|
| Idle after boot and embedder warmup | 404 MiB | 79% |
| After three sequential questions | 479 MiB | 94% |
| Peak during a six-way concurrent burst | 505 MiB | 99% |

Without the UI mounted the same measurements are 361 MiB idle, 417 MiB sequential, and
430 MiB peak, so Gradio costs roughly 60-75 MiB. Nothing OOM-killed in any test, but a
99% peak leaves about 7 MiB of headroom, which is thin for a public endpoint. Dropping
Gradio's event queue (`build_demo().queue()`) does not meaningfully change this. Use the
`standard` plan (2 GB) if the service will see real concurrent traffic; the `free` plan
is viable for a demo or portfolio piece.

Free instances spin down after 15 minutes idle, so the first request after a gap pays
roughly a 30 s cold start while the index verifies and the embedder loads.

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

## Known limitation: short "value" chunks are hard to retrieve

Some questions the corpus *can* answer are still refused. The refusal is truthful and
safe, but it is a real gap, not intended behaviour.

Scheme pages state facts as very short labelled rows. For `hdfc_elss` the index holds
three separate exit-load chunks, including a bare `Exit load\nNil` and
`Exit Load\n01 Jan 2013: --`. These are too short to sit near a full-sentence query, so
a question about them ranks the scheme's generic `About…` summary chunk first instead.

Two things in the query path make this worse:

- `rag/retriever.py` embeds the query *including the full scheme name*, even though
  the scheme is already applied as a Chroma metadata filter. The long fund name is
  noise that matches every chunk of that scheme equally and dilutes the real term.
- `rag/query_expand.py` appends synonym variants (`exit load` →
  `exit charge redemption charge exit penalty`). For these short value chunks the
  extra terms *dilute* rather than help: expansion measurably scores worse than not
  expanding at all.

Measured along the real production path (`expand()` then `embed_query()`), against the
`hdfc_elss` chunk holding the actual value:

| Query | As shipped | No expansion | Scheme name stripped |
|---|---|---|---|
| `exit load HDFC ELSS` | rank 3, 0.6670, fail | rank 1, 0.5294, pass | rank 1, 0.2141, pass |
| `Does HDFC ELSS have an exit load?` | rank 3, 0.6753, fail | rank 2, 0.5908, fail | rank 1, 0.3061, pass |
| `What is the exit load on HDFC ELSS Tax Saver Fund - Direct Plan - Growth?` | rank 10, 0.6886, fail | rank 11, 0.6751, fail | rank 1, 0.3172, pass |

Note the middle column: for the terse query, shipping the expander is *worse* than
not expanding (0.6670 versus 0.5294). Expansion is a net negative for this class of
question, not merely insufficient.

When this happens the model receives only the glossary definition of the term and
correctly returns the out-of-corpus sentinel, so `verify()` substitutes the static
refusal. Nothing is hallucinated; the system just fails to find a fact it holds.

This is a class of bug, not a single query: anything whose answer is `Nil`, `--`, or
otherwise absent-looking is affected.

Stripping the scheme name recovers all three cases at rank 1 and inside the existing
0.57 gate, with no recalibration. It is still not a safe blanket change, because the
remaining text can get too short to match — measured best-hit distance for
`What is the benchmark of HDFC Balanced Advantage Fund - Direct Growth?`:

| Query | As shipped | Scheme name stripped |
|---|---|---|
| Balanced Advantage "benchmark" | 0.1994, `About…`, pass | 0.7563, `Returns and rankings`, **fail** |

Stripping scheme aliases alone is safe but recovers nothing. Loosening `MAX_DISTANCE`
to 0.69 would admit the right chunk here, but it was set deliberately from
terse-query measurements and raising it weakens the guarantee the design exists to
provide.

The correct fix is dual-query retrieval: embed both the original query and a
scheme-stripped variant, merge and dedupe the hits, and let the existing gate judge
the merged best. That captures both wins above without touching the gate. Widening the
synonym list would make this worse, not better. Neither is implemented, so treat
short-value questions as unanswered for now. Reproduce with
`python -m rag.retriever --probe "<question>"`.


## Safety
Facts-only. No investment advice.

