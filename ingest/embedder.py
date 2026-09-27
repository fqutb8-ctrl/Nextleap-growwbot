from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

import config
from rag.logging_utils import get_logger

log = get_logger("embed")

_MODEL: object | None = None

PROBE_SAME_TOPIC = (
    "What is the expense ratio of HDFC Large Cap Fund?",
    "The expense ratio of HDFC Large Cap Fund Direct Growth is 1.03%.",
)
PROBE_UNRELATED = (
    "What is the expense ratio of HDFC Large Cap Fund?",
    "The Mughal Empire was founded in 1526 by Babur in northern India.",
)


class EmbeddingError(RuntimeError):
    pass


class ModelMismatchError(EmbeddingError):
    pass


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        log.info("model_loading", model=config.EMBED_MODEL, device=config.EMBED_DEVICE)
        model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
        reported = int(model.get_sentence_embedding_dimension())
        if reported != config.EMBED_DIM:
            raise EmbeddingError(
                f"{config.EMBED_MODEL} reports dim {reported} "
                f"but config.EMBED_DIM is {config.EMBED_DIM}"
            )
        if config.EMBED_MAX_SEQ_LENGTH != model.max_seq_length:
            log.warn(
                "max_seq_length_raised",
                model_default=model.max_seq_length,
                configured=config.EMBED_MAX_SEQ_LENGTH,
            )
            model.max_seq_length = config.EMBED_MAX_SEQ_LENGTH
        _MODEL = model
    return _MODEL


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _empty() -> np.ndarray:
    return np.zeros((0, config.EMBED_DIM), dtype=np.float32)


def _assert_shape(matrix: np.ndarray, expected_rows: int) -> None:
    expected = (expected_rows, config.EMBED_DIM)
    if matrix.shape != expected:
        raise EmbeddingError(f"expected embedding shape {expected}, got {matrix.shape}")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise EmbeddingError("model produced a zero-length vector")
    return (matrix / norms).astype(np.float32)


def _encode(texts: list[str], batch_size: int) -> np.ndarray:
    model = get_model()
    started = time.perf_counter()
    raw = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    matrix = _normalize(np.asarray(raw, dtype=np.float32))
    _assert_shape(matrix, len(texts))
    elapsed = time.perf_counter() - started
    log.info(
        "encoded",
        vectors=len(texts),
        batch=batch_size,
        t=f"{elapsed:.2f}s",
    )
    return matrix


def _cache_paths() -> tuple[Path, Path]:
    npy = config.EMBED_CACHE_NPY
    keys = config.EMBED_CACHE_KEYS_JSON
    return npy, keys


def _cache_identity() -> dict[str, object]:
    return {
        "model": config.EMBED_MODEL,
        "dim": config.EMBED_DIM,
        "max_seq_length": config.EMBED_MAX_SEQ_LENGTH,
    }


def _load_cache() -> tuple[dict[str, int], np.ndarray]:
    npy_path, keys_path = _cache_paths()
    if not (npy_path.is_file() and keys_path.is_file()):
        return {}, _empty()
    try:
        payload = json.loads(keys_path.read_text(encoding="utf-8"))
        matrix = np.load(npy_path)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        log.warn("cache_unreadable", reason=type(exc).__name__)
        return {}, _empty()

    identity = _cache_identity()
    if any(payload.get(key) != value for key, value in identity.items()):
        log.warn(
            "cache_discarded",
            cached_model=payload.get("model"),
            cached_window=payload.get("max_seq_length"),
            config_model=identity["model"],
            config_window=identity["max_seq_length"],
        )
        return {}, _empty()

    keys = payload.get("keys") or []
    if matrix.ndim != 2 or matrix.shape != (len(keys), config.EMBED_DIM):
        rows = int(matrix.shape[0]) if matrix.ndim == 2 else -1
        log.warn("cache_discarded", cached_rows=rows, cached_keys=len(keys))
        return {}, _empty()
    return {key: row for row, key in enumerate(keys)}, matrix.astype(np.float32)


def _save_cache(index: dict[str, int], matrix: np.ndarray) -> None:
    if not index:
        return
    npy_path, keys_path = _cache_paths()
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = [""] * len(index)
    for key, row in index.items():
        keys[row] = key

    keys_tmp = keys_path.with_name(f"{keys_path.stem}.tmp.json")
    keys_tmp.write_text(
        json.dumps({**_cache_identity(), "keys": keys}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(keys_tmp, keys_path)

    npy_tmp = npy_path.with_name(f"{npy_path.stem}.tmp.npy")
    with npy_tmp.open("wb") as handle:
        np.save(handle, matrix)
    os.replace(npy_tmp, npy_path)


def embed_texts(
    texts: Sequence[str], batch_size: int | None = None, use_cache: bool = True
) -> np.ndarray:
    items = list(texts)
    if not items:
        return _empty()
    for position, text in enumerate(items):
        if not text or not text.strip():
            raise EmbeddingError(f"texts[{position}] is blank")
    batch = config.EMBED_BATCH if batch_size is None else batch_size
    if batch < 1:
        raise EmbeddingError(f"batch size must be >= 1, got {batch}")

    keys = [_text_key(text) for text in items]
    index, matrix = _load_cache() if use_cache else ({}, _empty())

    hits = 0
    pending: list[int] = []
    unseen: set[str] = set()
    for position, key in enumerate(keys):
        if key in index:
            hits += 1
        elif key not in unseen:
            unseen.add(key)
            pending.append(position)

    log.info(
        "cache",
        requested=len(items),
        hits=hits,
        misses=len(unseen),
        rows_cached=len(index),
    )

    if pending:
        fresh = _encode([items[position] for position in pending], batch)
        for offset, position in enumerate(pending):
            key = keys[position]
            if key in index:
                continue
            index[key] = matrix.shape[0]
            matrix = np.vstack([matrix, fresh[offset : offset + 1]])
        if use_cache:
            _save_cache(index, matrix)

    out = np.vstack([matrix[index[key]] for key in keys])
    _assert_shape(out, len(items))
    return out


def embed_query(text: str) -> list[float]:
    if not text or not text.strip():
        raise EmbeddingError("query is blank")
    return [float(value) for value in _encode([text], 1)[0]]


def collection_metadata(built_at: str | None = None) -> dict[str, object]:
    stamp = built_at or (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    return {
        "hnsw:space": "cosine",
        "embed_model": config.EMBED_MODEL,
        "embed_dim": config.EMBED_DIM,
        "embed_max_seq_length": config.EMBED_MAX_SEQ_LENGTH,
        "corpus_built_at": stamp,
    }


def assert_model_compatible(collection_metadata: dict | None) -> None:
    if not collection_metadata:
        raise ModelMismatchError(
            f"collection metadata is empty; expected embed_model={config.EMBED_MODEL!r}"
        )
    found = str(collection_metadata.get("embed_model", "<missing>"))
    if found != config.EMBED_MODEL:
        raise ModelMismatchError(
            f"embedding model mismatch: collection embed_model={found!r} "
            f"vs config.EMBED_MODEL={config.EMBED_MODEL!r}; rebuild the index"
        )
    found_dim = collection_metadata.get("embed_dim")
    if found_dim is not None and int(found_dim) != config.EMBED_DIM:
        raise ModelMismatchError(
            f"embedding dim mismatch: collection embed_dim={found_dim} "
            f"vs config.EMBED_DIM={config.EMBED_DIM}; rebuild the index"
        )
    found_window = collection_metadata.get("embed_max_seq_length")
    if found_window is not None and int(found_window) != config.EMBED_MAX_SEQ_LENGTH:
        raise ModelMismatchError(
            f"embedding window mismatch: collection embed_max_seq_length={found_window} "
            f"vs config.EMBED_MAX_SEQ_LENGTH={config.EMBED_MAX_SEQ_LENGTH}; rebuild the index"
        )


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def probe() -> str:
    model = get_model()
    same = embed_texts(list(PROBE_SAME_TOPIC))
    unrelated = embed_texts(list(PROBE_UNRELATED))
    norms = np.linalg.norm(same, axis=1)
    sim_same = _similarity(same[0], same[1])
    sim_unrelated = _similarity(unrelated[0], unrelated[1])
    passed = sim_same > sim_unrelated
    status = "PASS" if passed else "FAIL"

    window = int(model.max_seq_length)
    lengths = [
        len(model.tokenizer.encode(a, add_special_tokens=False))
        + len(model.tokenizer.encode(b, add_special_tokens=False))
        for a, b in (PROBE_SAME_TOPIC, PROBE_UNRELATED)
    ]
    fits = max(lengths) <= window

    lines = [
        f"model       {config.EMBED_MODEL}",
        f"dim         {same.shape[1]} (config.EMBED_DIM={config.EMBED_DIM})",
        f"device      {config.EMBED_DEVICE}",
        f"window      {window} tokens (native default 256)",
        f"norms       min={norms.min():.6f} max={norms.max():.6f} (unit vectors)",
        "",
        "same topic:",
        f"  A  {PROBE_SAME_TOPIC[0]}",
        f"  B  {PROBE_SAME_TOPIC[1]}",
        f"  sim={sim_same:.4f}",
        "",
        "unrelated:",
        f"  A  {PROBE_UNRELATED[0]}",
        f"  B  {PROBE_UNRELATED[1]}",
        f"  sim={sim_unrelated:.4f}",
        "",
        f"assert sim(same topic) > sim(unrelated)  ->  {status}",
        f"assert probe pairs fit inside the {window}-token window  ->  "
        f"{'PASS' if fits else 'FAIL'} (longest pair {max(lengths)} tokens)",
    ]

    index, matrix = _load_cache()
    lines.append(f"cache rows  {len(index)} (dim {matrix.shape[1] if matrix.ndim == 2 else 0})")
    return "\n".join(lines)


def probe_corpus(sample: int) -> str:
    from ingest.chunker import count_tokens

    chunks = [
        json.loads(line)
        for line in config.CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    picked = chunks[:sample]
    matrix = embed_texts([chunk["text"] for chunk in picked])
    proxy = [count_tokens(chunk["text"]) for chunk in picked]
    model_tokens = [
        int(len(get_model().tokenizer.encode(chunk["text"], add_special_tokens=False)))
        for chunk in picked
    ]
    ratios = [m / p for m, p in zip(model_tokens, proxy) if p]
    return "\n".join(
        [
            f"corpus sample   {len(picked)} chunks from {config.CHUNKS_JSONL.name}",
            f"dim            {matrix.shape[1]}",
            f"norm min/max   {np.linalg.norm(matrix, axis=1).min():.6f} / "
            f"{np.linalg.norm(matrix, axis=1).max():.6f}",
            f"tiktoken proxy {min(proxy)}-{max(proxy)} tokens",
            f"MiniLM actual  {min(model_tokens)}-{max(model_tokens)} tokens",
            f"actual/proxy   mean={np.mean(ratios):.3f} min={min(ratios):.3f} max={max(ratios):.3f}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pinned sentence-transformer embedder with an on-disk vector cache."
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="print dim, vector norms, and a same-topic vs unrelated similarity check",
    )
    parser.add_argument(
        "--corpus",
        type=int,
        metavar="N",
        help="embed the first N real chunks and compare tokenizer counts",
    )
    args = parser.parse_args()

    if not args.probe and not args.corpus:
        parser.print_help()
        return 0

    try:
        if args.probe:
            print(probe())
        if args.corpus:
            if not config.CHUNKS_JSONL.is_file():
                log.error("missing_chunks", path=str(config.CHUNKS_JSONL))
                return 1
            print()
            print(probe_corpus(args.corpus))
    except (EmbeddingError, OSError) as exc:
        log.error("aborted", reason=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
