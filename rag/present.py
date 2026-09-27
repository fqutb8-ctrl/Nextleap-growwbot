from __future__ import annotations

import json
from typing import Any

import config
from ingest.embedder import EmbeddingError
from rag.logging_utils import get_logger
from rag.retriever import RetrievalError
from rag.schemas import Answer

log = get_logger("present")

BUILD_HINT = "The search index is unavailable. Run `python -m ingest.build_index` first."


def index_ready() -> bool:
    try:
        from rag.retriever import get_collection

        get_collection()
    except (RetrievalError, EmbeddingError):
        return False
    return True


def source_count() -> int:
    try:
        from ingest.loaders import load_sources

        return len(load_sources())
    except Exception as exc:
        log.error("source_count_failed", reason=f"{type(exc).__name__}: {exc}")
        return 0


def corpus_meta() -> dict[str, Any]:
    if not config.CORPUS_META_JSON.exists():
        return {}
    try:
        payload = json.loads(config.CORPUS_META_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.error("corpus_meta_unreadable", reason=str(exc))
        return {}
    return payload if isinstance(payload, dict) else {}


def corpus_line() -> str:
    meta = corpus_meta()
    parts = [f"Sources: {source_count()} scheme pages"]
    updated = meta.get("last_updated") or "unknown"
    parts.append(f"Last updated: {updated}")
    chunks = meta.get("chunk_count")
    if chunks:
        parts.append(f"Chunks: {chunks}")
    if not index_ready():
        parts.append(BUILD_HINT)
    return " · ".join(parts)


def trace_payload(result: Answer) -> dict[str, Any]:
    trace = result.trace
    return {
        "guards": trace.guards,
        "expanded_query": trace.expanded_query,
        "filter": trace.filter,
        "hits": trace.hits,
        "timings_ms": trace.timings_ms,
        "warnings": trace.warnings,
    }


def citation_payload(result: Answer) -> list[dict[str, str]]:
    return [
        {
            "source_url": citation.source_url,
            "section": citation.section,
            "scheme_name": citation.scheme_name,
            "fetched_at": citation.fetched_at,
        }
        for citation in result.citations
    ]
