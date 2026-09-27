from __future__ import annotations

import argparse

import config
from ingest.embedder import EmbeddingError, assert_model_compatible, embed_query
from rag.logging_utils import get_logger
from rag.query_expand import detect_scheme, expand
from rag.schemas import Hit

log = get_logger("retrieve")


class RetrievalError(RuntimeError):
    pass


class MissingCollectionError(RetrievalError):
    pass


class EmptyCollectionError(RetrievalError):
    pass


IndexMissingError = MissingCollectionError


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
        path = str(meta.get("heading_path", ""))
        hits.append(
            Hit(
                chunk_id=chunk_id,
                text=text or "",
                section=str(meta.get("section", "")),
                scheme=str(meta.get("scheme", "")),
                scheme_name=str(meta.get("scheme_name", "")),
                category=str(meta.get("category", "")),
                source_url=str(meta.get("source_url", "")),
                fetched_at=str(meta.get("fetched_at", "")),
                distance=float(distance),
                heading_path=[part for part in path.split(" > ") if part],
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


def gate_score(hits: list[Hit]) -> float:
    aggregation = getattr(config, "GATE_AGGREGATION", "min")
    distances = [hit.distance for hit in hits]
    if aggregation == "min":
        return min(distances)
    if aggregation == "max":
        return max(distances)
    raise RetrievalError(
        f"unknown GATE_AGGREGATION {aggregation!r}; expected 'max' or 'min'"
    )


def passes_gate(hits: list[Hit]) -> bool:
    if not hits:
        return False
    return gate_score(hits) <= config.MAX_DISTANCE


def format_probe(question: str, hits: list[Hit], scheme: str | None) -> str:
    aggregation = getattr(config, "GATE_AGGREGATION", "min")
    label = "best" if aggregation == "min" else "worst"
    score = gate_score(hits) if hits else None
    score_text = f"{score:.4f}" if score is not None else "n/a"
    verdict = "PASS" if passes_gate(hits) else "FAIL"
    lines = [
        f"question: {question}",
        f"expanded: {expand(question)}",
        f"scheme filter: {scheme or 'none'}",
        f"gate: {verdict} ({label}_distance="
        f"{score_text} limit={config.MAX_DISTANCE} aggregation={aggregation})",
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


def run_interactive(scheme: str | None, unfiltered: bool, show_text: int) -> int:
    from ingest.embedder import get_model

    get_model()
    print(f"retrieval tester | collection={config.COLLECTION_NAME} top_k={config.TOP_K}")
    print(
        f"gate: aggregation={getattr(config, 'GATE_AGGREGATION', 'min')} "
        f"limit={config.MAX_DISTANCE}"
    )
    print(f"scheme filter: {scheme or 'auto-detect'}{' (disabled)' if unfiltered else ''}")
    print("type a question, or 'quit' to exit\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            return 0
        try:
            hits = retrieve(question, scheme, auto_detect=not unfiltered)
            applied = None if unfiltered else (scheme or detect_scheme(question))
        except (RetrievalError, EmbeddingError) as exc:
            log.error("probe_failed", reason=str(exc))
            continue
        print()
        print(format_probe(question, hits, applied))
        if show_text:
            for rank, hit in enumerate(hits, 1):
                if rank > show_text:
                    print(f"   ... {len(hits) - show_text} more hit(s), rerun with --show-text {config.TOP_K}")
                    break
                print(f"   [{rank}] {hit.chunk_id} {hit.section}")
                print("   " + hit.text.replace("\n", " ")[:400])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe the vector index and print ranked hits with distances."
    )
    parser.add_argument("--probe", help="question to retrieve for, then exit")
    parser.add_argument(
        "--scheme",
        default=None,
        help="force a scheme filter instead of auto-detecting one",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="keep asking questions in one process instead of requiring --probe",
    )
    parser.add_argument(
        "--unfiltered",
        action="store_true",
        help="skip scheme auto-detection so all five schemes compete",
    )
    parser.add_argument(
        "--show-text",
        type=int,
        default=0,
        metavar="N",
        help="print the first N hit texts in full (interactive mode)",
    )
    args = parser.parse_args()

    if not args.probe and not args.interactive:
        parser.print_help()
        return 0

    if args.interactive:
        if args.show_text < 0:
            parser.error("--show-text must be >= 0")
        return run_interactive(args.scheme, args.unfiltered, args.show_text)

    try:
        hits = retrieve(args.probe, args.scheme, auto_detect=not args.unfiltered)
    except (RetrievalError, EmbeddingError) as exc:
        log.error("probe_failed", reason=str(exc))
        return 1

    applied = None if args.unfiltered else (args.scheme or detect_scheme(args.probe))
    print(format_probe(args.probe, hits, applied))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
