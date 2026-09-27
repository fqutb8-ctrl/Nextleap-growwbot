from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from ingest.embedder import embed_texts, get_model
from rag.logging_utils import get_logger

log = get_logger("dump")

RULE = "=" * 78
THIN = "-" * 78


def load_chunks() -> list[dict]:
    if not config.CHUNKS_JSONL.is_file():
        raise FileNotFoundError(f"missing {config.CHUNKS_JSONL}")
    return [
        json.loads(line)
        for line in config.CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def format_vector(vector: np.ndarray, per_line: int) -> str:
    lines = []
    for start in range(0, len(vector), per_line):
        window = vector[start : start + per_line]
        cells = " ".join(f"{float(value):+.4f}" for value in window)
        lines.append(f"  [{start:>3}] {cells}")
    return "\n".join(lines)


def nearest(matrix: np.ndarray, position: int, chunks: list[dict], count: int) -> str:
    if matrix.shape[0] < 2:
        return "  (none)"
    scores = matrix @ matrix[position]
    order = np.argsort(-scores)
    picked = [int(i) for i in order if int(i) != position][:count]
    if not picked:
        return "  (none)"
    lines = []
    for other in picked:
        distance = max(0.0, 1.0 - float(scores[other]))
        lines.append(
            f"  {float(scores[other]):.4f} (dist {distance:.4f})  "
            f"{chunks[other]['chunk_id']}  {chunks[other]['scheme']}  "
            f"{chunks[other]['section']}"
        )
    return "\n".join(lines)


def header(chunks: list[dict], matrix: np.ndarray) -> str:
    norms = np.linalg.norm(matrix, axis=1)
    similarities = matrix @ matrix.T
    np.fill_diagonal(similarities, -1.0)
    schemes = sorted({chunk["scheme"] for chunk in chunks})
    sections = sorted({chunk["section"] for chunk in chunks})
    return "\n".join(
        [
            RULE,
            "HDFC MUTUAL FUND RAG - CHUNKS AND EMBEDDINGS",
            RULE,
            f"generated        {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"source           {config.CHUNKS_JSONL.name}",
            f"embed model      {config.EMBED_MODEL}",
            f"device           {config.EMBED_DEVICE}",
            f"dimensions       {matrix.shape[1]} (config.EMBED_DIM={config.EMBED_DIM})",
            f"token window     {config.EMBED_MAX_SEQ_LENGTH} tokens",
            f"chunks           {len(chunks)}",
            f"schemes          {len(schemes)}  ({', '.join(schemes)})",
            f"sections         {len(sections)}",
            f"tokens total     {sum(int(c['n_tokens']) for c in chunks)}",
            f"tokens range     {min(int(c['n_tokens']) for c in chunks)}-"
            f"{max(int(c['n_tokens']) for c in chunks)}",
            f"vector norm      min={norms.min():.6f} max={norms.max():.6f} (unit vectors)",
            "",
            "Vectors are L2-normalized, so cosine similarity is the dot product and",
            "Chroma cosine distance is 1 - dot. Retrieval ranks by ascending distance",
            f"and rejects hits whose distance exceeds {config.MAX_DISTANCE}.",
            "Rounded to 4 decimals for reading; the index stores full float32 values.",
            "",
            "Per-chunk 'NEAREST' lists the 3 closest other chunks by cosine similarity",
            "computed from this same matrix, excluding the chunk itself.",
            "",
            RULE,
            "TABLE OF CONTENTS",
            RULE,
        ]
    )


def table_of_contents(chunks: list[dict], matrix: np.ndarray) -> str:
    lines = []
    for position, chunk in enumerate(chunks):
        preview = " ".join(chunk["text"].split())[:58]
        lines.append(
            f"{position + 1:>4}  {chunk['chunk_id']}  {chunk['n_tokens']:>3}tok  "
            f"{chunk['scheme']:<18}  {chunk['section'][:24]:<24}  {preview}"
        )
    lines.append("")
    lines.append(RULE)
    return "\n".join(lines)


def chunk_block(
    position: int,
    chunk: dict,
    vector: np.ndarray,
    matrix: np.ndarray,
    chunks: list[dict],
    per_line: int,
    neighbours: int,
    show_vectors: bool,
) -> str:
    heading = " > ".join(chunk.get("heading_path") or [chunk["section"]])
    lines = [
        RULE,
        f"CHUNK {position + 1}/{len(chunks)}   id={chunk['chunk_id']}",
        RULE,
        f"  scheme       {chunk['scheme']}  ({chunk['scheme_name']})",
        f"  category     {chunk['category']}",
        f"  section      {chunk['section']}",
        f"  heading path {heading}",
        f"  block types  {chunk['block_types']}",
        f"  n_tokens     {chunk['n_tokens']}   chars {len(chunk['text'])}",
        f"  source_url   {chunk['source_url']}",
        f"  fetched_at   {chunk['fetched_at']}",
        "",
        "TEXT",
        THIN,
        chunk["text"],
        "",
    ]
    if show_vectors:
        lines += [
            f"EMBEDDING  dim={len(vector)}  norm={float(np.linalg.norm(vector)):.6f}",
            THIN,
            format_vector(vector, per_line),
            "",
        ]
    if neighbours:
        lines += [f"NEAREST {neighbours}", THIN, nearest(matrix, position, chunks, neighbours), ""]
    return "\n".join(lines)


def build(out_path: Path, per_line: int, neighbours: int, show_vectors: bool, limit: int | None) -> str:
    chunks = load_chunks()
    if limit is not None:
        chunks = chunks[:limit]
    log.info("embedding_chunks", count=len(chunks))
    matrix = embed_texts([chunk["text"] for chunk in chunks])
    if matrix.shape != (len(chunks), config.EMBED_DIM):
        raise ValueError(f"unexpected embedding matrix shape {matrix.shape}")

    get_model()
    parts = [header(chunks, matrix), table_of_contents(chunks, matrix)]
    for position, chunk in enumerate(chunks):
        parts.append(
            chunk_block(
                position,
                chunk,
                matrix[position],
                matrix,
                chunks,
                per_line,
                neighbours,
                show_vectors,
            )
        )
    parts.append(RULE)
    parts.append("END OF DUMP")
    parts.append(RULE)
    text = "\n".join(parts) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump chunk text and embeddings to a human-readable text file."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.ARTIFACTS_DIR / "chunks_and_embeddings.txt",
        help="output text file path",
    )
    parser.add_argument("--limit", type=int, help="only dump the first N chunks")
    parser.add_argument("--per-line", type=int, default=8, help="vector values per line")
    parser.add_argument("--neighbours", type=int, default=3, help="nearest chunks to show")
    parser.add_argument(
        "--no-vectors", action="store_true", help="skip the numeric vectors, keep text only"
    )
    args = parser.parse_args()

    if args.per_line < 1:
        parser.error("--per-line must be >= 1")
    if args.neighbours < 0:
        parser.error("--neighbours must be >= 0")

    try:
        text = build(args.out, args.per_line, args.neighbours, not args.no_vectors, args.limit)
    except (FileNotFoundError, ValueError) as exc:
        log.error("aborted", reason=str(exc))
        return 1

    log.info("wrote", path=str(args.out), bytes=len(text.encode("utf-8")))
    print(f"wrote {args.out} ({len(text.encode('utf-8')):,} bytes, {text.count(chr(10)):,} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
