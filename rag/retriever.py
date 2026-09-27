from __future__ import annotations

import argparse
from dataclasses import dataclass

import config
from ingest.embedder import EmbeddingError, assert_model_compatible, embed_query
from rag.logging_utils import get_logger
from rag.query_expand import detect_scheme, expand

log = get_logger("retrieve")


class RetrievalError(RuntimeError):
    pass


class MissingCollectionError(RetrievalError):
    pass


class EmptyCollectionError(RetrievalError):
    pass


@dataclass
class Hit:
    chunk_id: str
    text: str
    section: str
    scheme: str
    source_url: str
    fetched_at: str
    distance: float


def get_collection():
    import chromadb

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    names = {entry.name for entry in client.list_collections()}
    if config.COLLECTION_NAME not in names:
        raise MissingCollectionError(
            f"collection {config.COLLECTION_NAME!r} not found in {config.CHROMA_DIR}; "
            "run python -m ingest.build_index first"
        )
    collection = client.get_collection(name=config.COLLECTION_NAME, embedding_function=None)
    assert_model_compatible(collection.metadata)
    if collection.count() == 0:
        raise EmptyCollectionError(
            f"collection {config.COLLECTION_NAME!r} exists but holds no chunks; "
            "run python -m ingest.build_index first"
        )
    return collection


def _to_hits(result) -> list[Hit]:
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    hits: list[Hit] = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        meta = meta or {}
        hits.append(
            Hit(
                chunk_id=chunk_id,
                text=text or "",
                section=str(meta.get("section", "")),
                scheme=str(meta.get("scheme", "")),
                source_url=str(meta.get("source_url", "")),
                fetched_at=str(meta.get("fetched_at", "")),
                distance=float(distance),
            )
        )
    return hits


def retrieve(question: str, scheme: str | None = None, auto_detect: bool = True) -> list[Hit]:
    collection = get_collection()
    if scheme is None and auto_detect:
        scheme = detect_scheme(question)
    if scheme is not None and scheme not in config.SCHEME_URLS:
        raise RetrievalError(
            f"unknown scheme {scheme!r}; expected one of {sorted(config.SCHEME_URLS)}"
        )

    expanded = expand(question)
    vector = embed_query(expanded)
    where = {"scheme": scheme} if scheme else None
    result = collection.query(
        query_embeddings=[vector],
        n_results=config.TOP_K,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = _to_hits(result)
    log.info(
        "retrieved",
        question=question,
        expanded=expanded,
        scheme=scheme or "none",
        filtered=where is not None,
        hits=len(hits),
        ids=",".join(hit.chunk_id for hit in hits),
        distances=",".join(f"{hit.distance:.4f}" for hit in hits),
    )
    return hits


def passes_gate(hits: list[Hit]) -> bool:
    if not hits:
        return False
    distances = [hit.distance for hit in hits]
    aggregation = getattr(config, "GATE_AGGREGATION", "max")
    if aggregation == "min":
        return min(distances) <= config.MAX_DISTANCE
    if aggregation == "max":
        return max(distances) <= config.MAX_DISTANCE
    raise RetrievalError(
        f"unknown GATE_AGGREGATION {aggregation!r}; expected 'max' or 'min'"
    )


def format_probe(question: str, hits: list[Hit], scheme: str | None) -> str:
    lines = [
        f"question: {question}",
        f"expanded: {expand(question)}",
        f"scheme filter: {scheme or 'none'}",
        f"gate: {'PASS' if passes_gate(hits) else 'FAIL'} "
        f"(max_distance={max((h.distance for h in hits), default=0.0):.4f} "
        f"limit={config.MAX_DISTANCE})",
        "",
    ]
    if not hits:
        lines.append("no hits")
        return "\n".join(lines)
    for rank, hit in enumerate(hits, 1):
        lines.append(
            f"{rank}. d={hit.distance:.4f} {hit.chunk_id} "
            f"[{hit.scheme}] {hit.section}"
        )
        lines.append(f"   {hit.source_url}")
        preview = " ".join(hit.text.split())[:160]
        lines.append(f"   {preview}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe the vector index and print ranked hits with distances."
    )
    parser.add_argument("--probe", required=True, help="question to retrieve for")
    parser.add_argument(
        "--scheme",
        default=None,
        help="force a scheme filter instead of auto-detecting one",
    )
    args = parser.parse_args()

    try:
        hits = retrieve(args.probe, args.scheme)
    except (RetrievalError, EmbeddingError) as exc:
        log.error("probe_failed", reason=str(exc))
        return 1

    applied = args.scheme or detect_scheme(args.probe)
    print(format_probe(args.probe, hits, applied))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
