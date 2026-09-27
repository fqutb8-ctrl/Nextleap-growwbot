# Implementation Guide — Mutual Fund FAQ Assistant (RAG Chatbot)

**Companion to:** `architecture.md` (design) · `PRD.md` (requirements)
**How to use:** Implement strictly in phase order. Each phase ends with a **Gate** — all gate checks must pass before starting the next phase.
**Status:** Draft v1.0 · 2026-09-27

---

## 0. How to Use This With Cursor

### 0.1 Working rules

1. **One phase at a time.** Never let Cursor implement two phases in one prompt.
2. **Paste the phase's "Cursor Prompt" verbatim**, then append any Cursor-specific context it asks for.
3. **Run the gate commands yourself** before moving on. Do not trust "done" — check the output.
4. **Commit after every passing gate** (commit messages provided per phase).
5. **Never let Cursor invent facts.** If a fact about HDFC schemes is needed for a test fixture, it must come from the fetched corpus, not from Cursor's memory. Instruct it: *"Do not hardcode scheme facts from memory; derive fixtures from artifacts/."*

### 0.2 Phase dependency graph

```
P0 Scaffold
 ├──▶ P1 Corpus + Fetch ──▶ P2 Structure Extract ──▶ P3 Chunker (+ analyze) ──┐
 │                                                                          │
 └──▶ P4 Embedder ──────────────────────────────────────────────┐            │
                                                                 ▼            ▼
                                                          P5 Vector Store / build_index
                                                                 │
                                                                 ▼
                                              P6 Guards ──▶ P7 Query Expand ──▶ P8 Retriever
                                                                                          │
                                                                                          ▼
                                                                        P9 Prompts + Chain ──▶ P10 Verify
                                                                                                  │
                                                                                                  ▼
                                                                                        P11 UI ──▶ P12 Eval
                                                                                                  │
                                                                                                  ▼
                                                                                        P13 Docs + Demo
```

### 0.3 Global Definition of Done

A phase is done when **all** hold:

- [ ] Every file listed in the phase exists with the listed signatures.
- [ ] Gate commands run clean from a fresh shell.
- [ ] `python -c "import <module>"` succeeds for each new module (no import-time side effects, no network calls at import).
- [ ] No secrets in code; no `.env` committed.
- [ ] Console logs include the `stage=` field for that stage (architecture §8).
- [ ] Phase committed with the provided message.

---

## Phase 0 — Project Scaffold & Configuration

**Goal:** Repo skeleton, dependencies, config, `.env` handling, git hygiene. Nothing functional.
**Refs:** architecture §3, §6 · PRD NFR1, NFR5
**Effort:** 0.25 day

### Tasks

- [ ] `P0.1` Create directory tree exactly as architecture §3 (`ingest/`, `rag/`, `eval/`, `artifacts/`, `tests/`)
- [ ] `P0.2` `requirements.txt` — pin minor versions: `chromadb`, `sentence-transformers`, `transformers`, `torch` (CPU), `httpx`, `beautifulsoup4`, `lxml`, `tiktoken`, `gradio`, `python-dotenv`, `pytest`
- [ ] `P0.3` `config.py` — every constant from architecture §6, no magic numbers elsewhere
- [ ] `P0.4` `.env.example` (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `HF_HOME`) and `.gitignore` (`.env`, `chroma_db/`, `artifacts/`, `__pycache__/`, `.venv/`)
- [ ] `P0.5` `rag/schemas.py` — dataclasses only, no logic, no imports beyond stdlib/typing
- [ ] `P0.6` `rag/logging_utils.py` — `get_logger(stage)` emitting `stage=... key=value` lines
- [ ] `P0.7` `README.md` skeleton with scope + disclaimer (PRD §8.4 text)
- [ ] `P0.8` `.python-version` / note Python 3.10+ in README

### `rag/schemas.py` — required types

```python
@dataclass
class Block:            kind: str; text: str          # kind: text|table|list|keyvalue
@dataclass
class Section:          heading: str; level: int; heading_path: list[str]; blocks: list[Block]
@dataclass
class Document:         doc_id: str; title: str; scheme: str; scheme_name: str
                        category: str; source_url: str; fetched_at: str
                        sections: list[Section]
@dataclass
class Chunk:            chunk_id: str; text: str; scheme: str; scheme_name: str
                        category: str; section: str; heading_path: list[str]
                        source_url: str; fetched_at: str; n_tokens: int
@dataclass
class Citation:         source_url: str; section: str; scheme_name: str; fetched_at: str
@dataclass
class QueryTrace:       guards: dict; expanded_query: str; filter: dict | None
                        hits: list[dict]; timings_ms: dict; warnings: list[str]
@dataclass
class Answer:           text: str; citations: list[Citation]; last_updated: str
                        trace: QueryTrace; refusal: bool; refusal_type: str | None
```

Also add `RefusalType` as a `str` Enum: `ADVICE`, `PII`, `PERFORMANCE`, `OUT_OF_CORPUS`.

### Gate

```bash
python -c "import config, rag.schemas, rag.logging_utils; print('ok')"
python -m pytest -q          # should collect 0 tests, exit 5 or 0
git status --short
```

### Commit

```
chore: scaffold project structure, config, and dataclass schemas
```

### Cursor Prompt

```
Read PRD.md and architecture.md. Implement Phase 0 only.

Create the directory tree from architecture.md section 3, a pinned
requirements.txt, config.py containing every constant listed in
architecture.md section 6 (no other magic numbers anywhere), .env.example,
.gitignore, README.md skeleton (must include the disclaimer text from
PRD section 8.4), and rag/schemas.py with exactly the dataclasses and fields
listed in Phase 0 of implementation.md.

Also create rag/logging_utils.py exposing get_logger(stage) which prints
"stage=<name> key=value" lines.

Constraints:
- No network calls at import time.
- No comments in code unless required for type clarity.
- Use pathlib, not os.path.
Do not implement any RAG logic yet.
```

---

## Phase 1 — Corpus Allowlist & Fetching

**Goal:** Read the 5 URLs from `ingest/sources.csv`, fetch them politely, cache raw HTML, and prove the pages are reachable.
**Refs:** architecture §4.1 · PRD FR1, §4.2, §4.4
**Effort:** 0.5 day
**Resolves open decision:** O4 (do these pages need headless rendering?)

### Tasks

- [ ] `P1.1` `ingest/sources.csv` — header `url,scheme,scheme_name,category`; 5 rows from PRD §4.2. Slugs: `hdfc_large_cap`, `hdfc_flexi_cap`, `hdfc_elss`, `hdfc_small_cap`, `hdfc_balanced_advantage`
- [ ] `P1.2` `ingest/loaders.py::load_sources()` → `list[dict]`, validates each URL is `https` and host in `{"groww.in"}`
- [ ] `P1.3` `fetch_url(url) -> FetchResult` — httpx, `timeout=20`, max 2 retries w/ backoff, 1.5 s inter-request delay, descriptive User-Agent, status + content-type + byte length in result
- [ ] `P1.4` Write raw HTML to `artifacts/raw_<scheme>.html`
- [ ] `P1.5` `fetch_all() -> list[FetchResult]` + CLI: `python -m ingest.loaders`
- [ ] `P1.6` Abort loudly on any non-200; never return a partial page as success
- [ ] `P1.7` Record `fetched_at` (UTC ISO 8601) per fetch, persist to `artifacts/fetch_manifest.json`

### Gate

```bash
python -m ingest.loaders
```

Expect: 5 successful fetches with byte counts in the 100 KB–1 MB range each, and 5 files in `artifacts/`. Then open one HTML file and confirm the scheme name and the words "expense ratio" are present.

```bash
python -c "from pathlib import Path; t=Path('artifacts/raw_hdfc_large_cap.html').read_text(encoding='utf-8'); print(len(t), 'expense ratio' in t.lower())"
```

### Decision gate O4

If extracted visible text (next phase) is < 500 chars, the page is JS-rendered → add optional `playwright` headless retry in `fetch_url` before proceeding. Record the finding in README known-limits.

### Pitfalls

- Groww may serve a consent/bot wall. If HTML contains "enable JavaScript" and no scheme data, that is a **blocking** issue — report it, do not attempt to bypass any access control.
- Always set an explicit `User-Agent`; default httpx UA is frequently blocked.

### Commit

```
feat(ingest): source allowlist and polite HTML fetcher with raw caching
```

### Cursor Prompt

```
Implement Phase 1 from implementation.md, using architecture.md section 4.1.

Create ingest/sources.csv with the 5 HDFC Groww URLs and slugs from PRD
section 4.2. Create ingest/loaders.py with load_sources(), fetch_url(), and
fetch_all() per Phase 1 task list, plus a __main__ CLI.

Rules:
- Allowlist enforcement: https only, host must be groww.in.
- 1.5s delay between requests, timeout 20s, 2 retries with backoff.
- Cache raw HTML to artifacts/raw_<scheme>.html.
- Write artifacts/fetch_manifest.json with url, scheme, status, bytes, fetched_at.
- Fail loudly on non-200. Never silently return partial content.
- No comments in code. No hardcoded URLs in Python; read them from the CSV.
Then run `python -m ingest.loaders` and show me the output.
```

---

## Phase 2 — Clean & Structure Extraction

**Goal:** Raw HTML → `Document` with a heading tree, tables preserved as Markdown, key-value pairs kept adjacent.
**Refs:** architecture §4.2, §4.3
**Effort:** 0.5 day

### Tasks

- [ ] `P2.1` `ingest/loaders.py::clean_html(html) -> str` — strip script/style/noscript/svg/iframe, then nav/header/footer/aside, cookie banners, breadcrumbs
- [ ] `P2.2` `extract_main_root(soup) -> Tag` — try `config.LOADER_CONTENT_SELECTORS` in order; fall back to largest-text-container heuristic
- [ ] `P2.3` `parse_document(fetch_result) -> Document` — walk headings (`h1`–`h4`) building `Section` nodes with `heading_path`
- [ ] `P2.4` `table_to_markdown(table) -> Block` — preserve header row + all rows
- [ ] `P2.5` `keyvalue_to_block(ul_or_dl) -> Block` — for "Minimum SIP: ₹500" style lists; never split label from value
- [ ] `P2.6` `load_documents() -> list[Document]` (cached from `artifacts/`) + `dump_documents()` → `artifacts/raw_docs.json`
- [ ] `P2.7` CLI `python -m ingest.loaders --parse` printing per-scheme section count + char count

### Gate

```bash
python -m ingest.loaders --parse
```

Expect per-scheme: ≥10 sections, sensible headings. Then inspect:

```bash
python -c "import json;d=json.load(open('artifacts/raw_docs.json'));x=[s['heading'] for s in d[0]['sections']];print(len(x));print(x[:20])"
```

Confirm headings include the fee/benchmark/riskometer-type sections for that scheme, and that nav junk ("Download app", "Cookie policy") is absent.

### Pitfalls

- Heading hierarchy on these pages is inconsistent. If `heading_path` is noisy, fall back to the nearest preceding heading of any level and note it in README.
- Do not lowercase or rewrite text at this stage — normalization happens in the chunker only.

### Commit

```
feat(ingest): boilerplate cleaning and heading-tree structure extraction
```

### Cursor Prompt

```
Implement Phase 2 from implementation.md (architecture section 4.2 and 4.3).

Add to ingest/loaders.py: clean_html(), extract_main_root(), parse_document(),
table_to_markdown(), keyvalue_to_block(), load_documents(), dump_documents(),
and a --parse CLI flag. Output artifacts/raw_docs.json matching the Document
dataclass in rag/schemas.py.

Critical rules:
- Tables become Markdown Blocks; never flatten a table into prose.
- A label and its value must stay in the same Block.
- No text rewriting, lowercasing, or summarization at this stage.
- Content root selection must be driven by config.LOADER_CONTENT_SELECTORS
  with a largest-text-container fallback.
- No comments in code.
Run `python -m ingest.loaders --parse` and print the per-scheme section counts
and the first 20 headings of the first document.
```

---

## Phase 3 — Chunker + Strategy Analysis

**Goal:** Produce `Chunk` objects and the instrumentation that decides the strategy.
**Refs:** architecture §4.4 · PRD §5.3
**Effort:** 0.5 day
**Resolves open decision:** O1

### Tasks

- [ ] `P3.1` `ingest/chunker.py::count_tokens(text) -> int` (tiktoken, single tokenizer instance)
- [ ] `P3.2` `chunk_by_heading(doc, max_tokens) -> list[Chunk]` — one section → one or more chunks
- [ ] `P3.3` `recursive_split(text, size, overlap, separators)` — separators ordered `["\n\n", "\n", ". ", " "]`
- [ ] `P3.4` `semantic_split(text, buffer_size=1, percentile=85, max_tokens)` — sentence embeddings, group by cosine breakpoints, then hard-cap
- [ ] `P3.5` `chunk_document(doc, strategy=config.CHUNK_STRATEGY) -> list[Chunk]` — dispatch; hybrid = heading → recursive for oversize → semantic for residuals
- [ ] `P3.6` `chunk_id = sha1(source_url + "|".join(heading_path) + ordinal)[:12]`
- [ ] `P3.7` Invariant assertions (raise, not warn): size cap respected, no cross-scheme chunk, non-empty `source_url` and `section`, tables split only on row boundaries
- [ ] `P3.8` `dump_chunks(chunks) -> artifacts/chunks.jsonl`
- [ ] `P3.9` `analyze(docs) -> table` printing the PRD §5.3 columns: `scheme, sections, sections>cap, avg_tokens, p90_tokens, table_sections, prose_sections`
- [ ] `P3.10` CLI `python -m ingest.chunker --analyze`

### Gate

```bash
python -m ingest.chunker --analyze
```

Then **make the decision** using the architecture §4.4 table, and write it down:

1. Copy the analyze output into README under "Chunking strategy rationale".
2. Set `config.CHUNK_STRATEGY` and `config.CHUNK_SIZE` / `CHUNK_OVERLAP` accordingly.
3. Justify in one paragraph referencing the observed numbers.

```bash
python -c "import json;cs=[json.loads(l) for l in open('artifacts/chunks.jsonl')];print(len(cs));import statistics as s;print('avg_tok',s.mean(c['n_tokens'] for c in cs),'max',max(c['n_tokens'] for c in cs))"
```

Expect 80–200 total chunks, avg 100–200 tokens, max ≤ `CHUNK_SIZE`.

Spot-check grounding: pick 3 chunks, and confirm the expense-ratio / exit-load / benchmark facts for one scheme are each fully contained in a single chunk with the URL in metadata.

### Pitfalls

- tiktoken is a *proxy* tokenizer, not MiniLM's. Use it for size budgeting only; store `n_tokens` as an estimate and label it so.
- Do not embed yet. Chunking must be re-tunable without re-embedding (architecture §4.4).

### Commit

```
feat(ingest): structure-aware chunker with semantic option and analysis CLI
```

### Cursor Prompt

```
Implement Phase 3 from implementation.md (architecture section 4.4).

Create ingest/chunker.py with count_tokens, chunk_by_heading, recursive_split,
semantic_split, chunk_document, dump_chunks, analyze, and a CLI supporting
`--analyze`.

Rules:
- Deterministic ids: sha1(source_url + heading_path + ordinal)[:12].
- Invariants must be enforced with raise, not warn: token cap, no cross-scheme
  chunk, non-empty source_url and section, tables split only on row boundaries.
- Do NOT embed anything. Output artifacts/chunks.jsonl only.
- No comments in code.
Then run `python -m ingest.chunker --analyze`, show me the table, and based on
architecture section 4.4 recommend the strategy with the numbers that justify
it. Do not set the config value yourself — recommend it and wait.
```

---

## Phase 4 — Embedder

**Goal:** One embedding entry point, model-pinned, cached, used by both ingestion and query.
**Refs:** architecture §4.5, §6
**Effort:** 0.25 day

### Tasks

- [ ] `P4.1` `ingest/embedder.py::get_model()` — cached singleton `SentenceTransformer(config.EMBED_MODEL)`
- [ ] `P4.2` `embed_texts(texts) -> np.ndarray` — batch `config.EMBED_BATCH`, L2-normalize, assert shape `(n, config.EMBED_DIM)`
- [ ] `P4.3` `embed_query(text) -> list[float]` — same model, separate name so the call site is explicit
- [ ] `P4.4` Disk cache: `artifacts/embed_cache.npy` + key index, keyed by `sha1(text)`; report hit/miss counts
- [ ] `P4.5` `assert_model_compatible(collection_metadata) -> None` — hard fail with both model names printed
- [ ] `P4.6` Record `EMBED_MODEL` + `EMBED_DIM` into collection metadata at build time
- [ ] `P4.7` CLI `python -m ingest.embedder --probe` printing dim, norm, and a 2-sentence similarity sanity check (same-topic pair should score higher than unrelated pair)

### Gate

```bash
python -m ingest.embedder --probe
```

Expect: dim 384, unit norms, and `sim(same topic) > sim(unrelated)`. First run downloads the model — allow ~2 min.

### Pitfalls

- First run needs network for the model download. Set `HF_HOME` in `.env` so it caches outside the repo.
- Never re-embed when only chunking changed. The cache is what makes Phase 3 iterations cheap.

### Commit

```
feat(ingest): pinned MiniLM embedder with disk cache and model-compat assertion
```

### Cursor Prompt

```
Implement Phase 4 from implementation.md (architecture section 4.5).

Create ingest/embedder.py with a cached get_model() singleton, embed_texts(),
embed_query(), an sha1-keyed disk cache at artifacts/embed_cache.npy, and
assert_model_compatible(). Add a --probe CLI.

Rules:
- Model and dim come from config.EMBED_MODEL / config.EMBED_DIM only.
- L2-normalize all vectors; assert output shape.
- The same model instance must serve ingestion and query time.
- No network calls at import; load the model lazily inside get_model().
- No comments in code.
Run `python -m ingest.embedder --probe` and show the output.
```

---

## Phase 5 — Vector Store & `build_index`

**Goal:** The one-command ingestion: CSV → documents → chunks → vectors → ChromaDB.
**Refs:** architecture §4.6, §4.7 · PRD FR1, FR2, NFR4
**Effort:** 0.5 day

### Tasks

- [ ] `P5.1` `ingest/build_index.py::get_client() -> chromadb.PersistentClient(path=config.CHROMA_DIR)`
- [ ] `P5.2` `build_collection(client, chunks, embeddings)` — `hnsw:space=cosine`, `embed_model`, `embed_dim`, `corpus_built_at` in metadata
- [ ] `P5.3` `upsert_chunks(...)` — `id=chunk_id`, `document=text`, `metadata={scheme, scheme_name, category, section, source_url, fetched_at, block_types}`
- [ ] `P5.4` Idempotency: delete + recreate collection when `config.REBUILD_COLLECTION` (default True)
- [ ] `P5.5` `corpus_last_updated() -> str` — max `fetched_at` across the collection; persisted to `artifacts/corpus_meta.json` for the UI footer
- [ ] `P5.6` `main()` orchestrating loaders → chunker → embedder → store, logging each stage per architecture §8
- [ ] `P5.7` CLI `python -m ingest.build_index`
- [ ] `P5.8` Verification helper `verify_index()` — count matches chunks, every metadata field non-empty, every `source_url` in the allowlist

### Gate

```bash
python -m ingest.build_index
python -m ingest.build_index          # run twice: counts must be identical (idempotency)
```

```bash
python -c "import chromadb;c=chromadb.PersistentClient(path='./chroma_db').get_collection('mf_faq_hdfc');print(c.count());print(c.metadata)"
```

Expect: count equal on both runs, `hnsw:space=cosine`, `embed_model` and `embed_dim=384` present.

Manual check — confirm the collection contains a chunk whose `section` mentions exit load and whose `source_url` is the Large Cap page.

### Pitfalls

- Chroma metadata values must be `str`, `int`, `float`, or `bool`. No `None`, no lists — `heading_path` must be joined to a string.
- Do not store the same chunk twice across runs; deterministic ids + recreate handles this.

### Commit

```
feat(ingest): build_index orchestrator writing chunks to persistent ChromaDB
```

### Cursor Prompt

```
Implement Phase 5 from implementation.md (architecture section 4.6 and 4.7).

Create ingest/build_index.py with get_client(), build_collection(),
upsert_chunks(), corpus_last_updated(), verify_index(), and main().

Rules:
- Chroma metadata values must be str/int/float/bool only; join heading_path
  into a string. No None values.
- Recreate the collection when config.REBUILD_COLLECTION is True.
- Log one line per stage with the stage= field, matching the examples in
  architecture section 8.
- No comments in code.
Run `python -m ingest.build_index` twice and show me both counts plus the
collection metadata.
```

---

## Phase 6 — Guardrails (advice, PII, performance)

**Goal:** All three refusals working as pure functions, before any retrieval work.
**Refs:** architecture §5.1–5.3, §5.11 · PRD §8.2, §8.3, FR7, FR9, FR10
**Effort:** 0.5 day

### Tasks

- [ ] `P6.1` `rag/prompts.py` — `ADVICE_REFUSAL`, `PII_REFUSAL`, `PERFORMANCE_REFUSAL`, `OUT_OF_CORPUS`, `DISCLAIMER` strings copied **verbatim** from PRD §8.4, plus `EDUCATION_LINK` and `factsheet_link(scheme)` helper
- [ ] `P6.2` `rag/guards.py::detect_advice_intent(q) -> bool` — regex list from architecture §5.1, compiled once at module load
- [ ] `P6.3` `rag/guards.py::detect_pii(q) -> list[str]` — pattern→name map from architecture §5.2; returns **names only**, never values
- [ ] `P6.4` `rag/guards.py::detect_performance(q) -> bool`
- [ ] `P6.5` `rag/guards.py::check(q) -> GuardResult` — single entry point returning `(action, refusal_type, matched_names)`, checking advice → pii → performance in that order
- [ ] `P6.6` `rag/guards.py::refusal_answer(refusal_type, scheme=None) -> Answer` — builds a static `Answer` with `trace` populated and **no LLM call**
- [ ] `P6.7` Query length cap (4000 chars) → truncate + warning
- [ ] `P6.8` `tests/test_guards.py` — 4 advice, 2 PII, 3 performance, 5 clean queries; assert refusal type and that the raw PII value never appears in `str(result)` or in captured logs

### Gate

```bash
python -m pytest tests/test_guards.py -v
```

All pass. Then verify the PII non-leak property explicitly:

```bash
python -c "from rag.guards import check,detect_pii;r=check('my pan is ABCDE1234F');print(detect_pii('my pan is ABCDE1234F'));print(r.action, 'ABCDE1234F' in str(r))"
```

Expect `['pan']`, action=refuse, and `False` for the leak check.

### Pitfalls

- A PAN-shaped string can appear inside a legitimate question. Accept the false positive — refusing is the correct trade-off (architecture §7).
- `str(result)` on any dataclass must never interpolate matched PII. If it does, fix the dataclass, not the test.

### Commit

```
feat(rag): advice, PII, and performance guardrails with static refusal answers
```

### Cursor Prompt

```
Implement Phase 6 from implementation.md (architecture sections 5.1 to 5.3).

Create rag/prompts.py with the refusal and disclaimer strings copied verbatim
from PRD section 8.4, plus EDUCATION_LINK and a factsheet_link() helper.
Create rag/guards.py with detect_advice_intent, detect_pii, detect_performance,
check(), and refusal_answer().

Rules:
- detect_pii returns pattern NAMES only, never the matched value.
- refusal_answer must construct an Answer with refusal=True and never call an
  LLM.
- Compile regexes once at module load.
- Check order: advice, then pii, then performance.
- No comments in code.
Create tests/test_guards.py covering 4 advice, 2 PII, 3 performance, and 5
clean queries, including an assertion that the raw PII value never appears in
the result or the logs. Run pytest and show the output.
```

---

## Phase 7 — Query Normalization & Expansion

**Goal:** Small, transparent query preprocessing.
**Refs:** architecture §5.4
**Effort:** 0.25 day

### Tasks

- [ ] `P7.1` `rag/query_expand.py::SYNONYMS` — the map from architecture §5.4, loaded from `config.QUERY_SYNONYMS` so it is editable without code changes
- [ ] `P7.2` `normalize(q) -> str` — whitespace + case normalization for matching; **original preserved for the LLM**
- [ ] `P7.3` `expand(q) -> str` — append synonym variants; cap expansion at ~2× original length
- [ ] `P7.4` `detect_scheme(q) -> str | None` — alias table (`elss`, `tax saver`, `large cap`, `flexi`, `small cap`, `balanced advantage`) → scheme slug; return `None` when zero or multiple match
- [ ] `P7.5` `tests/test_query_expand.py` — alias hits, multi-scheme ambiguity → `None`, no-match → `None`

### Gate

```bash
python -m pytest tests/test_query_expand.py -v
python -c "from rag.query_expand import expand,detect_scheme;print(expand('exit load'));print(detect_scheme('elss lock in period'), detect_scheme('compare large cap and small cap'))"
```

Expect synonyms appended; `hdfc_elss`; `None` for the ambiguous query.

### Pitfalls

- Over-eager synonym expansion hurts MiniLM retrieval. Keep the map small and log the expanded query so you can see its effect in the trace.

### Commit

```
feat(rag): query normalization, synonym expansion, and scheme detection
```

### Cursor Prompt

```
Implement Phase 7 from implementation.md (architecture section 5.4).

Create rag/query_expand.py with SYNONYMS (sourced from config.QUERY_SYNONYMS),
normalize(), expand(), and detect_scheme(). detect_scheme must return None when
zero or more than one scheme is mentioned.

Add tests/test_query_expand.py. No comments in code. Run pytest and show the
output, plus a live example of expand() and detect_scheme().
```

---

## Phase 8 — Retriever

**Goal:** Query → top-k chunks with scores, plus the weak-match score gate.
**Refs:** architecture §5.5–5.7, §4.7 · PRD FR4
**Effort:** 0.5 day

### Tasks

- [ ] `P8.1` `rag/retriever.py::get_collection()` — persistent client, asserts collection exists and `embed_model` matches `config.EMBED_MODEL` (architecture §7 startup check)
- [ ] `P8.2` `retrieve(question, scheme=None) -> list[Hit]` — embed via `ingest.embedder.embed_query`, `n_results=config.TOP_K`, `where={"scheme": scheme}` only when a single scheme was detected, `include=["documents","metadatas","distances"]`
- [ ] `P8.3` `Hit` dataclass: `chunk_id, text, section, scheme, source_url, fetched_at, distance`
- [ ] `P8.4` `passes_gate(hits) -> bool` — `max(distance) <= config.MAX_DISTANCE`
- [ ] `P8.5` Log every query: chunk ids + distances + filter applied (NFR7)
- [ ] `P8.6` Handle zero hits and empty collection distinctly
- [ ] `P8.7` CLI `python -m rag.retriever --probe "<question>"` printing ranked hits with distances — the demo's retrieval explainability tool
- [ ] `P8.8` `tests/test_retriever.py` — 10 known-answer questions, assert the expected scheme's page appears in top-k

### Gate

```bash
python -m rag.retriever --probe "what is the exit load on HDFC large cap direct growth"
python -m rag.retriever --probe "minimum SIP amount"
python -m rag.retriever --probe "lock in period for ELSS"
python -m pytest tests/test_retriever.py -v
```

Manually confirm: the exit-load probe's top hit is from `hdfc-large-cap-fund-direct-growth` with a section about exit load. The SIP probe should hit multiple schemes (proving the no-filter path works).

### Pitfalls

- `MAX_DISTANCE` is a placeholder (0.62). Do not tune it here — that is Phase 12 (open decision O2). Just make it configurable and log distances.
- Distance is cosine *distance* in Chroma, not similarity. Do not invert it twice.

### Commit

```
feat(rag): vector retriever with scheme filtering and weak-match score gate
```

### Cursor Prompt

```
Implement Phase 8 from implementation.md (architecture sections 5.5 to 5.7).

Create rag/retriever.py with get_collection() (asserting the collection exists
and embed_model matches config), a Hit dataclass, retrieve(),
passes_gate(), and a --probe CLI that prints ranked hits with distances.

Rules:
- Apply where={"scheme": ...} only when exactly one scheme was detected.
- Use ingest.embedder.embed_query so the model is guaranteed identical.
- Log chunk ids and distances on every query (NFR7).
- Handle zero hits and a missing collection as distinct errors.
- No comments in code.
Add tests/test_retriever.py with 10 known-answer questions asserting the
expected scheme page is in top-k. Run the three probe commands from Phase 8
and pytest, and show me all output.
```

---

## Phase 9 — Prompts, Context Building & Chain

**Goal:** The first end-to-end grounded answer with citation.
**Refs:** architecture §5.8, §5.10 · PRD §8.1, FR3, FR5, FR6
**Effort:** 0.5 day
**Resolves open decision:** O3 (LLM provider)

### Tasks

- [ ] `P9.1` Decide `config.LLM_PROVIDER` (ollama vs openai-compatible) and set `config.LLM_MODEL`; document the choice in README
- [ ] `P9.2` `rag/prompts.py::SYSTEM_PROMPT` — the 7 numbered rules from architecture §5.8, verbatim
- [ ] `P9.3` `rag/prompts.py::user_message(context_blocks, question) -> str` — context fenced, question fenced separately so it cannot be confused with context
- [ ] `P9.4` `rag/llm.py::generate(system, user) -> str` — provider adapter, `temperature=0`, `top_p=1`, single turn, no tools; one retry on transient failure then raise
- [ ] `P9.5` `rag/chain.py::build_context(hits) -> list[Block]` — heading path prefix, inline URL, dedupe at cosine > 0.97, drop lowest-scoring first under `config.CONTEXT_TOKEN_BUDGET`
- [ ] `P9.6` `rag/chain.py::answer(question) -> Answer` — the full ordered pipeline: guards → expand → retrieve → gate → context → generate → verify → decorate
- [ ] `P9.7` `rag/chain.py::last_updated_from(hits) -> str` — max `fetched_at`
- [ ] `P9.8` Populate `QueryTrace` at every step (guard decisions, expanded query, filter, hits + distances, timings_ms, warnings)
- [ ] `P9.9` `tests/test_chain.py` — monkeypatch `rag.llm.generate` to assert it is **not** called on any refusal path (architecture §5.11)

### Gate

```bash
python -c "from rag.chain import answer;a=answer('What is the expense ratio of HDFC Large Cap Fund Direct Growth?');print(a.text);print(a.citations);print(a.last_updated);print(a.trace.timings_ms)"
```

Expected: ≤3 sentences containing a real expense-ratio figure, exactly one citation whose `source_url` is the Large Cap page, and a `last_updated` date.

```bash
python -c "from rag.chain import answer;[print(answer(q).refusal_type, '|', answer(q).text[:70]) for q in ['Should I buy HDFC Large Cap?','Which is better, large cap or small cap?','My PAN is ABCDE1234F','What are the 5 year returns of HDFC Flexi Cap?','Who is the PM of HDFC AMC?']]"
```

Expected refusal types in order: `ADVICE`, `ADVICE`, `PII`, `PERFORMANCE`, `OUT_OF_CORPUS`.

```bash
python -m pytest tests/test_chain.py -v
```

### Pitfalls

- The out-of-scope question ("Who is the PM?") must be caught by the score gate or the model sentinel — **not** answered from model memory. If it answers, tighten the sentinel handling in Phase 10.
- The LLM may wrap the citation in markdown. That is fine; `Citation` objects are built from context metadata, not parsed out of the text.

### Commit

```
feat(rag): context builder, strict system prompt, and end-to-end answer chain
```

### Cursor Prompt

```
Implement Phase 9 from implementation.md (architecture sections 5.8 and 5.10).

Create rag/llm.py (provider adapter, temperature 0, single turn, one retry) and
finish rag/chain.py with build_context(), last_updated_from(), and answer().

The system prompt must contain the 7 numbered rules from architecture section
5.8 verbatim. Fence the context and the question separately.

answer() must run in this exact order: guards -> expand -> retrieve -> score
gate -> build context -> generate -> verify placeholder -> decorate. It must
return a fully populated Answer including QueryTrace with timings.

Rules:
- Refusal paths must never call rag.llm.generate.
- No comments in code.
Add tests/test_chain.py that monkeypatches rag.llm.generate to assert it is not
called on refusal paths. Then run the three gate commands from Phase 9 and
show me all output, including the printed trace for the expense-ratio question.
```

---

## Phase 10 — Post-Generation Verification & Decoration

**Goal:** Make the PRD's 100% guarantees deterministic.
**Refs:** architecture §5.9, §5.11 · PRD FR4, FR5, FR7, FR8, G2
**Effort:** 0.5 day

### Tasks

- [ ] `P10.1` `rag/chain.py::verify(text, hits) -> tuple[str, list[str]]` — returns possibly-corrected text + warnings
- [ ] `P10.2` Check: empty output or exact `OUT_OF_CORPUS` sentinel → return `OutOfCorpus` Answer
- [ ] `P10.3` Check: sentence count > 3 → keep first 3 sentences + warning
- [ ] `P10.4` Check: no context URL present in output → attach top-1 hit URL as citation + warning (citations must be 100%)
- [ ] `P10.5` Check: output contains advice or performance patterns → replace with the static refusal
- [ ] `P10.6` Check: output contains a URL not in context → strip it + warning
- [ ] `P10.7` `decorate(answer) -> Answer` — append `Last updated from sources: <date>`, ensure exactly one `Citation`
- [ ] `P10.8` `count_sentences(text) -> int` — abbreviation-safe (`e.g.`, `i.e.`, `Rs.`, `No.`)
- [ ] `P10.9` `tests/test_verify.py` — one test per check, each with a deliberately bad LLM stub: 5-sentence output, output with an invented URL, output containing "I recommend", output with no URL

### Gate

```bash
python -m pytest tests/test_verify.py tests/test_chain.py -v
```

Every stubbed-bad case must produce a safe `Answer` and a non-empty `trace.warnings` entry.

### Pitfalls

- Sentence counting is the classic source of off-by-one here. Test `"e.g. this is one sentence."` explicitly.
- Truncation must never produce a dangling fragment. Trim to the last complete sentence.

### Commit

```
feat(rag): deterministic post-generation verification and answer decoration
```

### Cursor Prompt

```
Implement Phase 10 from implementation.md (architecture section 5.9).

Add verify(), decorate(), and count_sentences() to rag/chain.py and implement
all six checks in the order listed there. Every failed check must add a message
to trace.warnings. Citation must end up with exactly one entry on every
answered response.

count_sentences must not split on abbreviations like "e.g.", "i.e.", "Rs.",
"No.".

Add tests/test_verify.py with a stubbed generate() for each failure mode:
5 sentences, invented URL, advice language, no URL at all, empty string.
Run pytest and show the output.
```

---

## Phase 11 — UI

**Goal:** Demoable chat: welcome, 3 examples, disclaimer, citations, trace expander.
**Refs:** PRD FR11, §6.1, §8.4
**Effort:** 0.5 day
**Resolves open decision:** O5 (Gradio vs Streamlit)

### Tasks

- [ ] `P11.1` Choose framework; default Gradio (architecture D12)
- [ ] `P11.2` `app.py` — chat interface wired to `rag.chain.answer`
- [ ] `P11.3` Header: welcome line + **"Facts-only. No investment advice."**
- [ ] `P11.4` The 3 PRD §6.1 example questions as clickable examples
- [ ] `P11.5` Render `Answer.citations` as clickable links below each response
- [ ] `P11.6` Render "Last updated from sources: …" in a muted footer
- [ ] `P11.7` "Show retrieval trace" expander: guards, expanded query, filter, chunk ids, distances, timings, warnings
- [ ] `P11.8` Show corpus last-updated date and source count on startup
- [ ] `P11.9` Handle `IndexMissingError` with the message "run `python -m ingest.build_index`"
- [ ] `P11.10` Never echo the user's raw question back if it contained PII (refusal responses only)

### Gate

```bash
python app.py
```

Manual checklist:
- [ ] Welcome line + disclaimer visible on load
- [ ] 3 example questions visible and clickable
- [ ] A factual question returns ≤3 sentences with a working citation link
- [ ] An advice question returns the refusal with the education link
- [ ] A PII question returns the refusal and the PAN is not visible anywhere
- [ ] Trace expander shows chunk ids and distances
- [ ] Corpus last-updated date is displayed

### Pitfalls

- Gradio streaming can reorder the citation; render citations as a separate component rather than inside the streamed text.
- Do not put the LLM key in the UI code.

### Commit

```
feat(ui): Gradio chat with citations, disclaimer, and retrieval trace
```

### Cursor Prompt

```
Implement Phase 11 from implementation.md.

Create app.py using Gradio (architecture decision D12). It must render:
- welcome line and the exact disclaimer "Facts-only. No investment advice."
- the 3 example questions from PRD section 6.1 as clickable examples
- each answer with exactly one clickable citation link
- the "Last updated from sources: <date>" line
- a collapsible "Show retrieval trace" expander fed by Answer.trace
- the corpus last-updated date and number of sources

Handle a missing index by telling the user to run
`python -m ingest.build_index`. Never display a raw question that triggered a
PII refusal.

No comments in code. Start the app and tell me the local URL.
```

---

## Phase 12 — Eval Harness & Threshold Calibration

**Goal:** Prove the PRD §10 metrics, and calibrate `MAX_DISTANCE` (O2).
**Refs:** PRD §10, architecture §10
**Effort:** 0.5 day

### Tasks

- [ ] `P12.1` `eval/test_questions.json` — 15–20 items: `{id, question, expected_schemes[], category, expect_refusal|expect_answer, expected_source_url?, notes}`; cover all 8 in-scope categories × ≥2 schemes, 4 advice traps, 2 PII traps, 3 performance traps, 2 out-of-corpus
- [ ] `P12.2` `eval/run_eval.py` — runs the set, prints a results table, writes `eval/results.json`
- [ ] `P12.3` Metrics: factual accuracy (human-verifiable field `verified_correct`), citation present + resolvable, refusal rate per trap class, ≤3-sentence rate, out-of-corpus decline rate
- [ ] `P12.4` **Calibrate `MAX_DISTANCE`:** run the answerable set and the unanswerable set, pick the threshold that separates them, write the value + the numbers into README
- [ ] `P12.5` Sweep `TOP_K` ∈ {5, 8} and record the accuracy delta in README (PRD §5.5)
- [ ] `P12.6` Save `eval/snapshots/` of 10 answers as a regression baseline
- [ ] `P12.7` Fix the worst 2 failures, re-run, record before/after

### Gate

```bash
python -m eval.run_eval
```

| Metric | Target |
|---|---|
| Factual accuracy | ≥90% |
| Citation present & resolvable | 100% |
| Advice refusals | 4/4 |
| PII refusals | 2/2 |
| Performance refusals | 3/3 |
| Out-of-corpus declines | 2/2 |
| Answers ≤3 sentences | 100% |

Then confirm the calibrated values are written into `config.py` and justified in README.

### Pitfalls

- Do not "fix" a failing factual answer by hardcoding the answer. Fix retrieval (chunking, k, threshold) or the prompt.
- If a fact genuinely is not on the 5 pages, the correct outcome is `OUT_OF_CORPUS`, not a guess. Reclassify the test item and note it.

### Commit

```
test(eval): golden-set harness, threshold calibration, and regression snapshots
```

### Cursor Prompt

```
Implement Phase 12 from implementation.md.

Create eval/test_questions.json (15-20 items covering the 8 in-scope categories
across at least 2 schemes, 4 advice traps, 2 PII traps, 3 performance traps,
2 out-of-corpus) and eval/run_eval.py that prints a results table and writes
eval/results.json.

The harness must compute: factual accuracy, citation present + resolvable,
refusal rate per trap class, <=3-sentence rate, and out-of-corpus decline rate.

Then run it, and report the numbers against the PRD section 10 targets. Also
run a TOP_K sweep of 5 vs 8 and report the accuracy delta. Do NOT hardcode any
scheme facts into the harness — everything must come from the corpus.
```

---

## Phase 13 — Documentation & Demo

**Goal:** All six PRD §11 deliverables finished.
**Refs:** PRD §11, §12, §13
**Effort:** 0.5 day

### Tasks

- [ ] `P13.1` README: setup (5 commands), scope (HDFC + 5 schemes), architecture diagram, **chunking rationale with the real analyze numbers**, calibrated `MAX_DISTANCE`, `TOP_K` sweep result, known limits, disclaimer
- [ ] `P13.2` `sources.csv` finalized and referenced from README (FR12)
- [ ] `P13.3` `SAMPLE_QA.md` — 8 real Q&A transcripts with answers + links, copied from `eval/results.json`
- [ ] `P13.4` `DISCLAIMER.md` — exact UI strings (PRD §8.4)
- [ ] `P13.5` `DEMO_SCRIPT.md` — the ≤3-min narration (below)
- [ ] `P13.6` Verify every "known limit" in README is real, not aspirational
- [ ] `P13.7` Final full run from a clean clone to prove NFR1

### Demo script (3 minutes)

| Time | Show |
|---|---|
| 0:00–0:20 | App loads: welcome, 3 examples, "Facts-only. No investment advice." |
| 0:20–0:50 | Ask expense ratio for Large Cap → answer + citation; open trace expander → show chunk ids and distances |
| 0:50–1:20 | Ask ELSS lock-in → answer + citation |
| 1:20–1:50 | Ask "Should I buy HDFC Large Cap?" → refusal + education link |
| 1:50–2:10 | Ask a PII question → refusal, PAN not echoed |
| 2:10–2:30 | Ask "5-year returns?" → performance refusal + factsheet link |
| 2:30–2:50 | In a terminal: `--analyze` output, then `build_index` stage logs |
| 2:50–3:00 | `eval/results.json` metrics table |

### Gate

```bash
git clone <repo> demo-check && cd demo-check
# follow README setup exactly
python -m ingest.build_index && python app.py
```

All five commands work with no undocumented step.

### Commit

```
docs: README, sample Q&A, disclaimer, and 3-minute demo script
```

### Cursor Prompt

```
Implement Phase 13 from implementation.md.

Fill in README.md using ONLY real numbers already produced by earlier phases:
the chunker --analyze table, the calibrated MAX_DISTANCE, the TOP_K sweep, and
the eval metrics. Do not invent or estimate any figure — if a number is missing,
leave a clearly marked TODO for me.

Create SAMPLE_QA.md from eval/results.json (8 real Q&A with links),
DISCLAIMER.md with the verbatim PRD 8.4 strings, and DEMO_SCRIPT.md following
the 3-minute table in Phase 13.

Do not claim any known limit in the README that we have not actually observed.
```

---

## Appendix A — Phase Checklist Table

| Phase | Deliverable | Gate | Commit | Est |
|---|---|---|---|---|
| 0 | Scaffold, config, schemas | imports clean | `chore: scaffold...` | 0.25 d |
| 1 | `sources.csv` + fetcher | 5 pages fetched | `feat(ingest): source allowlist...` | 0.5 d |
| 2 | Cleaner + structure extraction | ≥10 sections/scheme | `feat(ingest): boilerplate...` | 0.5 d |
| 3 | Chunker + `--analyze` | 80–200 chunks, strategy decided | `feat(ingest): structure-aware chunker...` | 0.5 d |
| 4 | Embedder | dim 384, sane similarity | `feat(ingest): pinned MiniLM embedder...` | 0.25 d |
| 5 | `build_index` → ChromaDB | idempotent, metadata correct | `feat(ingest): build_index orchestrator...` | 0.5 d |
| 6 | 3 guardrails | all refusal tests pass | `feat(rag): advice, PII, performance guardrails...` | 0.5 d |
| 7 | Query expansion | tests pass | `feat(rag): query normalization...` | 0.25 d |
| 8 | Retriever + gate | correct scheme in top-k | `feat(rag): vector retriever...` | 0.5 d |
| 9 | Prompts + chain | end-to-end cited answer | `feat(rag): context builder...` | 0.5 d |
| 10 | Verification | all bad-stub tests safe | `feat(rag): deterministic verification...` | 0.5 d |
| 11 | UI | manual checklist | `feat(ui): Gradio chat...` | 0.5 d |
| 12 | Eval + calibration | PRD §10 targets met | `test(eval): golden-set harness...` | 0.5 d |
| 13 | Docs + demo | clean-clone run works | `docs: README, sample Q&A...` | 0.5 d |
| | | | **Total** | **~5.75 d** |

## Appendix B — Open Decisions Tracker

| ID | Question | Phase | Status |
|---|---|---|---|
| O1 | Chunk strategy + `CHUNK_SIZE` | 3 | ☐ open |
| O4 | Do Groww pages need headless rendering? | 1 | ☐ open |
| O3 | LLM provider | 9 | ☐ open |
| O5 | Gradio vs Streamlit | 11 | ☐ open |
| O2 | `MAX_DISTANCE` calibration | 12 | ☐ open |
| O6 | Official factsheet link source for performance refusals | 6 | ☐ open |

## Appendix C — Global Constraints (paste into any Cursor prompt)

```
- Do not add comments to code unless required for type clarity.
- No magic numbers: all tunables live in config.py.
- No network calls at import time.
- No secrets in code; read LLM_API_KEY from the environment.
- Log one line per stage with a stage= field.
- Never hardcode scheme facts (expense ratios, exit loads, SIP amounts,
  lock-in periods, benchmarks, riskometer levels) from memory. All such facts
  must come from the ingested corpus.
- Never fabricate a URL. Citations come only from retrieved chunk metadata.
- If a fact is missing from the corpus, return OUT_OF_CORPUS. Do not guess.
- Match the dataclasses in rag/schemas.py and the constants in config.py.
```
