from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# corpus
SOURCES_CSV = BASE_DIR / "ingest" / "sources.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RAW_HTML_DIR = ARTIFACTS_DIR / "raw"
FETCH_MANIFEST = ARTIFACTS_DIR / "fetch_manifest.json"
RAW_DOCS_JSON = ARTIFACTS_DIR / "raw_docs.json"
CHUNKS_JSONL = ARTIFACTS_DIR / "chunks.jsonl"
COLLECTION_NAME = "mf_faq_hdfc"
CHROMA_DIR = BASE_DIR / "chroma_db"
REBUILD_COLLECTION = True

# fetching
HTTP_TIMEOUT_S = 20.0
HTTP_MAX_RETRIES = 2
HTTP_BACKOFF_S = 1.5
HTTP_DELAY_S = 1.5
HTTP_USER_AGENT = (
    "mf-faq-assistant/1.0 (educational class project; "
    "contact: student@example.edu)"
)
ALLOWED_URL_SCHEMES = ("https",)
ALLOWED_HOSTS = ("groww.in",)
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
MIN_HTML_CHARS = 500

# cleaning
LOADER_CONTENT_SELECTORS = (
    "div.layout-main",
    "div.layout-container",
    "main",
    "[role=main]",
    "div.container div.row div.col-md-8",
)
LOADER_MIN_ROOT_CHARS = 200
LOADER_DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "canvas",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "select",
    "option",
)
LOADER_BOILERPLATE_SELECTORS = (
    "[class*=cookie]",
    "[id*=cookie]",
    "[class*=consent]",
    "[class*=breadcrumb]",
    "[class*=newsletter]",
    "[class*=subscribe]",
    "[class*=appBanner]",
    "[class*=modalFooter]",
)
LOADER_JUNK_ELEMENT_TEXTS = (
    "see all",
    "view all",
    "view details",
    "show all",
    "show more",
    "download app",
    "open app",
    "get the app",
    "cookie policy",
    "accept all",
    "accept cookies",
    "got it",
    "skip to content",
)
LOADER_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
LOADER_MAX_HEADING_LEVEL = 5
LOADER_HEADING_PATH_MODE = "flat"
LOADER_OVERVIEW_HEADING = "Fund overview"
LOADER_RISK_PILL_PATTERN = r"(?i)\brisk\b"
LOADER_RISK_PILL_LABEL = "Riskometer level"
LOADER_KV_MAX_LABEL_CHARS = 30
LOADER_KV_REJECT_LABEL_PATTERN = r"\d+(?:\.\d+)?\s*%"
LOADER_KV_REJECT_NUMERIC_LABEL_PATTERN = r"^[+\-₹$]?\s*\d[\d.,]*\s*%?$"
LOADER_INTERACTIVE_CLASSES = ("cur-po",)
LOADER_WHITESPACE_PATTERN = r"\s+"

# embedding
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
EMBED_BATCH = 64
EMBED_CACHE_NPY = ARTIFACTS_DIR / "embed_cache.npy"
EMBED_CACHE_KEYS_JSON = ARTIFACTS_DIR / "embed_cache_keys.json"
EMBED_DEVICE = "cpu"
EMBED_MAX_SEQ_LENGTH = 512

# chunking
CHUNK_STRATEGY = "recursive"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEMANTIC_BREAKPOINT_PERCENTILE = 85
SEMANTIC_BUFFER_SIZE = 1
CHUNK_SEPARATORS = ("\n\n", "\n", ". ", " ")
CHUNK_INCLUDE_HEADING = True
TOKENIZER_ENCODING = "cl100k_base"
CHUNK_STRATEGIES = ("recursive", "semantic", "hybrid")

# retrieval
TOP_K = 5
MAX_DISTANCE = 0.62
CONTEXT_TOKEN_BUDGET = 1800

# llm
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = 0.0
LLM_API_KEY_ENV = "LLM_API_KEY"

QUERY_SYNONYMS = {
    "sip": ["systematic investment plan", "monthly investment"],
    "lock in": ["lock-in", "holding period", "lock-in period"],
    "ter": ["total expense ratio", "expense ratio"],
    "exit load": ["exit charge", "redemption charge", "exit penalty"],
    "riskometer": ["risk level", "risk category", "riskometer level"],
    "direct plan": ["direct growth", "direct growth plan"],
}

SCHEME_ALIASES = {
    "hdfc_large_cap": ["large cap", "largecap"],
    "hdfc_flexi_cap": ["flexi cap", "flexicap", "hdfc equity fund"],
    "hdfc_elss": ["elss", "tax saver", "tax saver fund"],
    "hdfc_small_cap": ["small cap", "smallcap"],
    "hdfc_balanced_advantage": ["balanced advantage", "balanced advantage fund"],
}


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
