# HDFC Mutual Fund FAQ — grounded RAG assistant

A retrieval-augmented assistant that answers questions about five HDFC Mutual Fund
Direct Growth schemes using only facts scraped from a fixed allowlist of Groww
scheme pages. No investment advice, no performance figures, no model-memory answers.

## Status

Phases 0–9 implemented. Phases 10–13 remain (post-generation verification, UI,
eval calibration, documentation).

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

## LLM provider (open decision O3)

**Chosen: Groq, `llama-3.3-70b-versatile`, over Groq's OpenAI-compatible API.**

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
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_KEY=<your key>
```

`LLM_API_KEY` is required for every hosted provider. `rag/llm.py` raises
`LLMNotConfiguredError` rather than silently degrading when the key is empty.
Ollama and LM Studio run without a key.

## Running

```bash
pip install -r requirements.txt

python -m ingest.build_index          # fetch, chunk, embed, persist, verify
python -m ingest.build_index --verify # verify the existing index only

python -m rag.retriever --probe "what is the exit load on HDFC large cap"
python -c "from rag.chain import answer; print(answer('what is the expense ratio of HDFC Large Cap Fund?').text)"

python -m pytest tests/ -v
```

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
