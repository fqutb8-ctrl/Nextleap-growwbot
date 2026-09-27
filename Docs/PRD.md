# PRD — Mutual Fund FAQ Assistant (RAG Chatbot)

**Type:** Class demo prototype
**Owner:** Team (HDFC AMC scope)
**Status:** Draft v1.0
**Last updated:** 2026-09-27

---

## 1. Problem Statement

Retail users comparing mutual fund schemes repeatedly ask the same factual questions — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer level, benchmark, and how to download statements. These answers already exist on public scheme pages, factsheets, and AMC/SEBI/AMFI guides, but they are scattered, inconsistently formatted, and hard to compare across schemes.

Support and content teams answer these questions manually, repeatedly. Users meanwhile get opinions ("should I buy?") dressed up as facts, with no source.

**We will build a facts-only RAG chatbot** that answers mutual fund scheme questions using only official public pages, cites exactly one source link per answer, and politely refuses advice-seeking questions.

---

## 2. Goals & Non-Goals

### 2.1 Goals

| # | Goal | Measure |
|---|------|---------|
| G1 | Answer factual MF scheme questions grounded strictly in retrieved corpus text | 100% of factual answers contain no claim absent from sources |
| G2 | Every answer carries exactly one citation link | 100% of answers |
| G3 | Refuse opinionated / portfolio questions politely + redirect to an educational link | 100% refusal on advice-intent test set |
| G4 | Demonstrate the full RAG lifecycle (ingestion **and** retrieval) end to end | Pipeline runs from raw URL → cited answer in one command |
| G5 | Answers are short and transparent about staleness | ≤3 sentences + "Last updated from sources: <date>" |

### 2.2 Non-Goals (out of scope)

- No performance calculations, return comparisons, or ranking of schemes.
- No buy/sell/hold recommendations, asset allocation, or portfolio advice.
- No login, authentication, user accounts, or personalization.
- No live NAV, real-time pricing, or scheme recommendation engine.
- No PII collection of any kind (see §8).
- No crawling beyond the 5 whitelisted pages + cited official documents (e.g. HDFC factsheet PDF) discovered *from* those pages.

---

## 3. Users

| Persona | Need | Success moment |
|---|---|---|
| Retail investor comparing HDFC schemes | "What's the exit load on HDFC Large Cap?" | Gets the number plus the official link, no opinion |
| First-time ELSS investor | "Is there a lock-in?" | Gets 3 years + section 80C context + link |
| Support / content team member | "What's the minimum SIP across these 5?" | Answer is ≤3 sentences, copy-pasteable, cited |

---

## 4. Scope

### 4.1 AMC

**HDFC Asset Management** — one AMC only, to keep the corpus small and quality high.

### 4.2 Schemes (5, all Direct – Growth)

| Category | Scheme | Source URL |
|---|---|---|
| Large Cap | HDFC Large Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| Flexi Cap | HDFC Equity (Flexi Cap) Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| ELSS | HDFC ELSS Tax Saver Fund – Direct Plan – Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| Small Cap | HDFC Small Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

### 4.3 Question Categories In Scope

1. Expense ratio / TER
2. Exit load
3. Minimum SIP (and minimum lumpsum)
4. Lock-in period (ELSS-specific) + tax treatment basics
5. Riskometer level / category
6. Benchmark
7. How to download capital-gains / account statements
8. Direct vs Regular plan difference (factual, not advisory)

### 4.4 Source Policy

- **Public official sources only**: Groww scheme pages (primary corpus), HDFC AMC factsheets / KIM / SID / scheme FAQ pages, SEBI and AMFI pages.
- **No third-party blogs, news articles, YouTube, or social content** as retrieval sources.
- No screenshots of any back-end/admin UI.
- If a needed fact is not in the corpus, the bot must say so and link the official page to check — never fill the gap from model memory.

---

## 5. Architecture

The system is split into two explicit pipelines. Both must be demonstrable.

### 5.1 Pipeline A — Data Ingestion (offline, run once per source change)

```
Source URLs (5 whitelisted)
   │
   ▼
[1] LOADING          Fetch HTML → clean boilerplate (nav/footer/scripts)
   │                 → extract main content text + section headings
   │                 → record: url, scheme, category, fetched_at
   ▼
[2] CHUNKING         Structure-aware split (§5.3)
   │                 → each chunk keeps: scheme, section, source_url, heading_path
   ▼
[3] EMBEDDING        HuggingFace sentence-transformers
   │                 → all-MiniLM-L6-v2 (384-dim, CPU-friendly)
   │                 → embed each chunk, store vector + payload metadata
   ▼
[4] VECTOR STORE     ChromaDB (persistent client)
                     → collection: mf_faq_hdfc
                     → metadata: scheme, category, section, url, fetched_at, chunk_id
```

### 5.2 Pipeline B — Query / Retrieval (online, per user question)

```
User question
   │
   ▼
[5] GUARDRAIL CHECK  Advice-intent classifier (§8.2)
   │                 → if ADVICE → polite refusal + educational link, STOP
   ▼
[6] PII CHECK        Redact/deny PAN, Aadhaar, account no., OTP, email, phone
   │                 → if PII detected → refuse, do not log the value
   ▼
[7] QUERY EXPANSION  Optional: synonym map (SIP ↔ systematic investment,
   │                 lock-in ↔ holding period, TER ↔ expense ratio)
   ▼
[8] EMBED QUERY      same all-MiniLM-L6-v2 model (MUST match ingestion model)
   ▼
[9] RETRIEVAL        ChromaDB similarity search, top-k = 5
   │                 → optional: metadata filter on scheme when detected
   ▼
[10] CONTEXT BUILD   Take top chunks, cap token budget, preserve heading context
   ▼
[11] GENERATION      LLM call with strict system prompt (§8.1)
   │                 → facts only, ≤3 sentences, one citation, no advice
   ▼
[12] POST-PROCESS    Verify a source_url exists in returned context
                     → append "Last updated from sources: <max fetched_at>"
                     → attach citation link to UI
   ▼
Answer + citation
```

### 5.3 Chunking Strategy — Decision Rule

**Decision is made by inspecting the actual ingested data, not assumed.** The team will run the ingestion script with instrumentation and decide:

| Signal in the data | Chosen strategy |
|---|---|
| Pages have clean, consistent section headings (expense ratio, exit load, benchmark, riskometer…) and each section is self-contained | **Recursive / structure-aware splitting** — split on headings first, then recursive character splitting with `chunk_size ≈ 400–600` tokens, `chunk_overlap ≈ 50–80` tokens |
| Sections are long, run-on, or a single fact is spread across paragraphs (e.g. fee tables embedded in prose) | **Semantic chunking** — embed consecutive sentences, group by similarity breakpoints, then apply a max-size cap |
| Mixed (most likely for these pages) | **Hybrid:** heading-based split → recursive split for oversized sections → semantic split only on sections exceeding the size cap |

**Requirements regardless of strategy:**
- Never split a table row or a label from its value ("Expense ratio 1.5%" must stay in one chunk).
- Never split across scheme boundaries — every chunk belongs to exactly one scheme.
- Every chunk carries `source_url` + `section` metadata so citations are always resolvable.
- Log the chunk-count and average chunk-size distribution per scheme to justify the final choice in the demo.

### 5.4 Embedding Model

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Dimensions:** 384
- **Rationale:** free, fast on CPU, strong baseline for short factual passages — appropriate for a class demo.
- **Constraint:** the exact same model + revision must be used at ingestion and at query time. Record the model name in the ChromaDB collection metadata and assert it at startup.

### 5.5 Vector Database

- **ChromaDB**, persistent client, local disk directory (e.g. `./chroma_db`).
- **Collection:** one per corpus (`mf_faq_hdfc`).
- **Similarity:** cosine.
- **Stored per chunk:** `id`, `document` (text), `embedding`, `metadata {scheme, category, section, source_url, heading_path, fetched_at, chunk_id}`.
- **Retrieval params:** `n_results = 5`; if answer quality is weak, try `n_results = 8` with a similarity threshold and report the tuning in the README.

### 5.6 LLM Layer

- Provider: pluggable. Any chat LLM available to the team (hosted API or local via Ollama).
- **Temperature: 0** (deterministic, factual).
- Input = system prompt (§8.1) + retrieved context (with URLs inline) + user question.
- If the LLM provider requires an API key, it is read from an environment variable and **never** committed to the repo.

---

## 6. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Ingestion script fetches all 5 whitelisted URLs, cleans them, chunks, embeds, and persists to ChromaDB. Re-runnable; idempotent (re-ingest replaces collection). |
| FR2 | Ingestion records `fetched_at` per document and exposes the corpus's latest date for the "Last updated" line. |
| FR3 | Chat endpoint answers a natural-language question using retrieval only. |
| FR4 | Every factual answer includes exactly one clickable source link from the corpus. |
| FR5 | Answers are ≤3 sentences. |
| FR6 | Every answer ends with "Last updated from sources: &lt;date&gt;". |
| FR7 | Advice-seeking questions ("Should I buy X?", "Is X good for me?", "Which is better, A or B?") are refused with a polite facts-only message + one relevant educational link. |
| FR8 | Out-of-corpus questions are answered with an explicit "not in my sources" message + the official page to check. The bot must not answer from model memory. |
| FR9 | PII-bearing input is refused and the value is never stored or logged. |
| FR10 | Performance/return questions ("returns of X vs Y", "which gave best SIP") are refused and redirected to the official factsheet link. |
| FR11 | UI shows a welcome line, 3 example questions, and the note "Facts-only. No investment advice." |
| FR12 | Source list is maintained as a machine-readable file (`sources.csv` or `sources.md`) and shown in the repo README. |

### 6.1 Example Questions (must appear in UI)

1. What is the expense ratio of the HDFC Large Cap Fund – Direct Growth?
2. Is there a lock-in period in the HDFC ELSS Tax Saver Fund?
3. What is the minimum SIP amount for HDFC Small Cap Fund – Direct Growth?

---

## 7. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR1 | Full pipeline reproducible from a clean clone with documented setup steps (≤5 commands). |
| NFR2 | Runs on CPU; no GPU required. |
| NFR3 | Answer latency target: <10 s per question on a laptop. |
| NFR4 | Ingestion is idempotent and re-runnable; corpus rebuild <5 min. |
| NFR5 | No secrets in the repository; `.env` git-ignored. |
| NFR6 | Code organized so each RAG stage is a separate, inspectable module (see §9). |
| NFR7 | Every stage logs enough to debug a wrong answer (retrieved chunk IDs + scores in console). |

---

## 8. Safety, Guardrails & Compliance

### 8.1 System Prompt Requirements (verbatim constraints)

- "You are a factual assistant for HDFC mutual fund scheme information."
- "Answer ONLY from the provided context. If the context does not contain the answer, say so."
- "Maximum 3 sentences."
- "Do not give investment advice, recommendations, opinions, or comparisons of expected returns."
- "Do not compute or state returns, CAGR, or performance figures."
- "End with: 'Last updated from sources: <date from context>'."
- "Cite exactly one source URL from the context."

### 8.2 Advice-Intent Guardrail

Keyword/heuristic classifier (fast, deterministic, demo-friendly) that flags:
`should I, which is better, is it good, recommend, suggest, worth buying, buy or sell, allocate, portfolio for me, best scheme, can I invest in`

On flag → return fixed refusal + one educational link (e.g. HDFC "Understanding Mutual Funds" / SEBI investor education page). Do **not** call the LLM for the refusal.

### 8.3 PII Guardrail

Detect and refuse: PAN (`[A-Z]{5}[0-9]{4}[A-Z]`), Aadhaar (12-digit), 10-digit phone numbers, email addresses, account numbers, OTPs, folio numbers. Never log the raw matched value — log only the pattern type.

### 8.4 Mandatory Disclaimers

- UI footer: **"Facts-only. No investment advice."**
- Refusal message: **"I can share public facts about this scheme, but I can't give investment advice or recommendations. Please consult a SEBI-registered investment advisor for guidance."**
- Performance refusal: **"I don't share return or performance figures. Please check the official factsheet for audited performance."**
- README must carry the same disclaimer.

---

## 9. Module Structure (each RAG stage separately inspectable)

```
.
├── ingest/
│   ├── loaders.py        # Stage 1: fetch + clean + extract
│   ├── chunker.py        # Stage 2: structure-aware / recursive / semantic
│   ├── embedder.py       # Stage 3: all-MiniLM-L6-v2
│   └── build_index.py    # Stage 4: write to ChromaDB
├── rag/
│   ├── guards.py         # Stage 5, 6: advice + PII
│   ├── retriever.py      # Stage 8, 9: embed query + Chroma search
│   ├── prompts.py        # system prompt
│   └── chain.py          # Stage 10-12: context build, generate, post-process
├── app.py                # Tiny UI (Gradio or Streamlit)
├── sources.csv           # FR12 source list
├── eval/
│   └── test_questions.json + runner   # §10
├── requirements.txt
├── .env.example
└── README.md
```

---

## 10. Evaluation & Success Criteria

Build a small golden set of **15–20 questions** covering: the 8 in-scope categories × at least 2 schemes, plus 4 advice-intent traps, plus 2 PII traps, plus 2 out-of-corpus questions.

| Metric | Target |
|---|---|
| Factual accuracy vs. source text | ≥90% on the in-scope golden set |
| Citation present & resolvable | 100% |
| Advice questions correctly refused | 100% (4/4) |
| PII questions correctly refused | 100% (2/2) |
| Out-of-corpus questions correctly declined | 100% (2/2) |
| Answers ≤3 sentences | 100% |
| Human rating of answer helpfulness (1–5) | ≥4 average |

Run the eval script, save results into the sample Q&A deliverable.

---

## 11. Deliverables

1. **Working prototype** — app link (or local run instructions) / notebook, or a ≤3-min demo video if hosting isn't possible.
2. **Source list** — `sources.csv` / `sources.md` with the 5 URLs used.
3. **README** — setup steps, scope (AMC + schemes), architecture diagram, chunking-strategy rationale, known limits.
4. **Sample Q&A file** — 5–10 queries with the assistant's answers + links.
5. **Disclaimer snippet** — exact text used in the UI.
6. **Architecture walkthrough** — demo narration of every RAG stage (loading → chunking → embedding → vector store → retrieval → generation), with logs shown for each.

---

## 12. Milestones

| # | Milestone | Deliverable | Est. |
|---|---|---|---|
| M1 | Corpus collection | 5 pages fetched + `sources.csv` | 0.5 day |
| M2 | Ingestion pipeline | Load → chunk → embed → ChromaDB working, chunk stats logged | 1 day |
| M3 | Chunking strategy decision | Instrumentation output + written rationale in README | 0.5 day |
| M4 | Retrieval + generation | RAG chain returning grounded answers with citations | 1 day |
| M5 | Guardrails | Advice refusal, PII refusal, out-of-corpus decline, ≤3-sentence + last-updated enforcement | 0.5 day |
| M6 | UI | Welcome + 3 examples + disclaimer + citation links | 0.5 day |
| M7 | Eval + tuning | Golden set run, results table, fixes | 0.5 day |
| M8 | Docs + demo | README, sample Q&A, demo video/script | 0.5 day |

**Total: ~5 days.**

---

## 13. Risks & Known Limits

| Risk / Limit | Mitigation |
|---|---|
| Groww page structure changes → cleaner breaks | Pin the extraction selectors; fall back to generic readability extraction; document in README |
| Groww pages are JS-rendered → text may not be in raw HTML | Use a headless fetch step if needed; **do not** scrape anything behind a login |
| Facts on pages go stale (expense ratio changes) | Show "Last updated from sources" date; re-run ingestion weekly; state the limitation in README |
| Small corpus (5 pages) → weak retrieval on unseen questions | Out-of-corpus decline path; expand corpus only from official HDFC/SEBI/AMFI docs |
| LLM may drift into advice tone | Temperature 0, strict system prompt, post-generation keyword check, refusal path bypasses LLM |
| "Last updated" derived from fetch date, not source's own as-of date | Label it precisely as the fetch date; do not claim it reflects the fund's official NAV date |
| MiniLM is English + short-passage oriented | Corpus is English and factual — acceptable; note as a limit |

---

## 14. Out of Scope for v1 (Possible v2)

- Multiple AMCs (Aditya Birla, ICICI Prudential, etc.).
- Direct HDFC AMC factsheet PDFs as an additional source type.
- Semantic/keyword hybrid search + reranking.
- Multilingual (Hindi/Hinglish) queries.
- Ambition: scheme comparison table generation (facts only, still no returns).
