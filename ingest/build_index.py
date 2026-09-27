from __future__ import annotations

import argparse
import json
import time
from urllib.parse import urlparse

import numpy as np

import config
from ingest.chunker import ChunkingError, chunk_documents
from ingest.embedder import EmbeddingError, collection_metadata, embed_texts
from ingest.loaders import StructureError, load_documents
from rag.logging_utils import get_logger
from rag.schemas import Chunk

log_load = get_logger("load")
log_chunk = get_logger("chunk")
log_embed = get_logger("embed")
log_persist = get_logger("persist")
log_verify = get_logger("verify")

REQUIRED_METADATA = (
    "scheme",
    "scheme_name",
    "category",
    "section",
    "heading_path",
    "source_url",
    "fetched_at",
    "block_types",
)


class IndexBuildError(RuntimeError):
    pass


class IndexVerificationError(IndexBuildError):
    pass


def get_client():
    import chromadb

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(client, create: bool = False):
    name = config.COLLECTION_NAME
    if create:
        return client.get_or_create_collection(
            name=name, metadata=collection_metadata(), embedding_function=None
        )
    collection = client.get_collection(name=name, embedding_function=None)
    attached = getattr(collection, "_embedding_function", None)
    if attached is not None:
        raise IndexBuildError(
            f"collection {name!r} resolved an embedding function ({type(attached).__name__}); "
            "query vectors must come from ingest.embedder, not the collection default"
        )
    return collection


def build_collection(client, chunks: list[Chunk], embeddings, built_at: str | None = None):
    name = config.COLLECTION_NAME
    existing = {entry.name for entry in client.list_collections()}
    if name in existing and config.REBUILD_COLLECTION:
        client.delete_collection(name)
        existing.discard(name)
        log_persist.info("collection_deleted", collection=name)
    if name in existing:
        raise IndexBuildError(
            f"collection {name!r} already exists and config.REBUILD_COLLECTION is False"
        )
    collection = client.get_or_create_collection(
        name=name,
        metadata=collection_metadata(built_at),
        embedding_function=None,
    )
    if len(chunks) != len(embeddings):
        raise IndexBuildError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    return collection


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    return {
        "scheme": chunk.scheme,
        "scheme_name": chunk.scheme_name,
        "category": chunk.category,
        "section": chunk.section,
        "heading_path": " > ".join(chunk.heading_path),
        "source_url": chunk.source_url,
        "fetched_at": chunk.fetched_at,
        "block_types": chunk.block_types,
        "n_tokens": int(chunk.n_tokens),
    }


def upsert_chunks(collection, chunks: list[Chunk], embeddings) -> int:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.shape != (len(chunks), config.EMBED_DIM):
        raise IndexBuildError(
            f"embedding matrix {matrix.shape} does not match "
            f"{len(chunks)} chunks at dim {config.EMBED_DIM}"
        )
    metadatas = [_metadata(chunk) for chunk in chunks]
    for chunk, meta in zip(chunks, metadatas):
        for key, value in meta.items():
            if not isinstance(value, (str, int, float, bool)):
                raise IndexBuildError(
                    f"chunk {chunk.chunk_id} metadata {key} is {type(value).__name__}, "
                    "not str/int/float/bool"
                )
            if isinstance(value, str) and not value:
                raise IndexBuildError(f"chunk {chunk.chunk_id} metadata {key} is empty")
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=metadatas,
        embeddings=matrix.tolist(),
    )
    return len(chunks)


def corpus_last_updated(collection=None) -> str:
    if collection is None:
        collection = get_collection(get_client())
    stored = collection.get(include=["metadatas"])
    stamps = [
        (meta or {}).get("fetched_at", "")
        for meta in stored.get("metadatas") or []
    ]
    stamps = [stamp for stamp in stamps if stamp]
    latest = max(stamps) if stamps else ""
    payload = {
        "collection": config.COLLECTION_NAME,
        "last_updated": latest,
        "chunk_count": collection.count(),
    }
    config.CORPUS_META_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.CORPUS_META_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_persist.info("corpus_meta", path=str(config.CORPUS_META_JSON), last_updated=latest)
    return latest


def _allowed_urls() -> set[str]:
    from ingest.loaders import load_sources

    return {row.url for row in load_sources()}


def verify_index(chunks: list[Chunk] | None = None) -> bool:
    collection = get_collection(get_client())
    problems: list[str] = []
    count = collection.count()
    expected = len(chunks) if chunks is not None else count
    if count != expected:
        problems.append(f"count is {count}, expected {expected}")

    stored = collection.get(include=["metadatas", "documents"])
    ids = stored.get("ids") or []
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    if len({*ids}) != len(ids):
        problems.append("duplicate ids in collection")

    allowed = _allowed_urls()
    for chunk_id, document, meta in zip(ids, documents, metadatas):
        meta = meta or {}
        for key in REQUIRED_METADATA:
            if key not in meta:
                problems.append(f"{chunk_id}: missing metadata {key}")
            elif meta[key] in ("", None):
                problems.append(f"{chunk_id}: empty metadata {key}")
        if not document:
            problems.append(f"{chunk_id}: empty document")
        url = meta.get("source_url", "")
        if url and url not in allowed:
            problems.append(f"{chunk_id}: source_url {url!r} is not in sources.csv")
        parsed = urlparse(url)
        if parsed.scheme not in config.ALLOWED_URL_SCHEMES:
            problems.append(f"{chunk_id}: scheme {parsed.scheme!r} not allowed")
        if parsed.hostname not in config.ALLOWED_HOSTS:
            problems.append(f"{chunk_id}: host {parsed.hostname!r} not allowed")

    if chunks is not None:
        expected_ids = {chunk.chunk_id for chunk in chunks}
        if set(ids) != expected_ids:
            missing = sorted(expected_ids - set(ids))
            extra = sorted(set(ids) - expected_ids)
            problems.append(f"id mismatch: {len(missing)} missing, {len(extra)} extra")

    for problem in problems[:20]:
        log_verify.error("problem", detail=problem)
    if problems:
        raise IndexVerificationError(
            f"index verification failed with {len(problems)} problem(s); "
            f"first: {problems[0]}"
        )
    log_verify.info("ok", chunks=count, fields=len(REQUIRED_METADATA), urls=len(allowed))
    return True


def prune_orphan_segments() -> list[str]:
    import shutil
    import sqlite3
    import uuid

    sqlite_path = config.CHROMA_DIR / "chroma.sqlite3"
    if not sqlite_path.is_file():
        return []
    try:
        connection = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
        try:
            rows = connection.execute("select id from segments").fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        log_persist.warn("prune_skipped", reason=str(exc))
        return []

    registered = {row[0] for row in rows}
    removed: list[str] = []
    for entry in sorted(config.CHROMA_DIR.iterdir()):
        if not entry.is_dir():
            continue
        try:
            uuid.UUID(entry.name)
        except ValueError:
            continue
        if entry.name in registered:
            continue
        freed = sum(item.stat().st_size for item in entry.rglob("*") if item.is_file())
        shutil.rmtree(entry)
        removed.append(f"{entry.name}:{freed}")
    if removed:
        log_persist.info("orphan_segments_removed", count=len(removed), ids=",".join(removed))
    else:
        log_persist.info("orphan_segments_none", registered=len(registered))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load, chunk, embed and persist the corpus into ChromaDB."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the existing collection against the chunk file and exit",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="delete vector-segment directories that the registry no longer references, then exit",
    )
    args = parser.parse_args()

    try:
        if args.verify:
            verify_index()
            return 0
        if args.prune:
            prune_orphan_segments()
            return 0

        started = time.perf_counter()
        documents = load_documents()
        log_load.info(
            "done",
            documents=len(documents),
            sections=sum(len(d.sections) for d in documents),
            t=f"{time.perf_counter() - started:.2f}s",
        )

        started = time.perf_counter()
        chunks = chunk_documents(documents)
        tokens = [chunk.n_tokens for chunk in chunks]
        log_chunk.info(
            "done",
            strategy=config.CHUNK_STRATEGY,
            chunks=len(chunks),
            avg_tok=int(sum(tokens) / len(tokens)),
            max_tok=max(tokens),
            t=f"{time.perf_counter() - started:.2f}s",
        )

        started = time.perf_counter()
        embeddings = embed_texts([chunk.text for chunk in chunks])
        log_embed.info(
            "done",
            model=config.EMBED_MODEL,
            vectors=int(embeddings.shape[0]),
            window=config.EMBED_MAX_SEQ_LENGTH,
            t=f"{time.perf_counter() - started:.2f}s",
        )

        started = time.perf_counter()
        client = get_client()
        collection = build_collection(client, chunks, embeddings)
        upserted = upsert_chunks(collection, chunks, embeddings)
        log_persist.info(
            "done",
            collection=config.COLLECTION_NAME,
            upserted=upserted,
            total=collection.count(),
            t=f"{time.perf_counter() - started:.2f}s",
        )

        corpus_last_updated(collection)
        verify_index(chunks)
        prune_orphan_segments()
    except (ChunkingError, EmbeddingError, IndexBuildError, StructureError) as exc:
        log_persist.error("aborted", reason=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
