from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import tiktoken

import config
from ingest.loaders import StructureError, load_documents
from rag.logging_utils import get_logger
from rag.schemas import Chunk, Document, Section

log = get_logger("chunk")

_ENCODING: tiktoken.Encoding | None = None
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_ROW = re.compile(r"^\s*\|.*\|\s*$")


class ChunkingError(RuntimeError):
    pass


class ChunkInvariantError(ChunkingError):
    pass


def _encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding(config.TOKENIZER_ENCODING)
    return _ENCODING


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text, disallowed_special=()))


def _hard_split(text: str, size: int) -> list[str]:
    encoding = _encoding()
    tokens = encoding.encode(text, disallowed_special=())
    return [
        encoding.decode(tokens[index : index + size])
        for index in range(0, len(tokens), size)
    ]


def _split_on(text: str, separator: str) -> list[str]:
    parts = text.split(separator)
    if len(parts) == 1:
        return [text]
    pieces = [part + separator for part in parts[:-1]] + [parts[-1]]
    return [piece for piece in pieces if piece.strip()]


def _split_to_pieces(text: str, size: int, separators: Sequence[str]) -> list[str]:
    if count_tokens(text) <= size:
        return [text] if text.strip() else []
    if not separators:
        return _hard_split(text, size)
    separator, rest = separators[0], separators[1:]
    parts = _split_on(text, separator)
    if len(parts) <= 1:
        return _split_to_pieces(text, size, rest)
    pieces: list[str] = []
    for part in parts:
        pieces.extend(_split_to_pieces(part, size, rest))
    return pieces


def _overlap_tail(pieces: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    tail: list[str] = []
    for piece in reversed(pieces):
        candidate = [piece] + tail
        if count_tokens("".join(candidate)) > overlap:
            break
        tail = candidate
    return tail


def _pack(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and count_tokens("".join(current + [piece])) > size:
            chunks.append("".join(current))
            current = _overlap_tail(current, overlap)
            if current and count_tokens("".join(current + [piece])) > size:
                current = []
        current.append(piece)
    if current:
        chunks.append("".join(current))
    return chunks


def recursive_split(
    text: str,
    size: int | None = None,
    overlap: int | None = None,
    separators: Sequence[str] | None = None,
) -> list[str]:
    limit = config.CHUNK_SIZE if size is None else size
    span = config.CHUNK_OVERLAP if overlap is None else overlap
    seps = tuple(config.CHUNK_SEPARATORS) if separators is None else tuple(separators)
    if not text.strip():
        return []
    pieces = _split_to_pieces(text, limit, seps)
    for piece in pieces:
        if count_tokens(piece) > limit:
            raise ChunkInvariantError(f"recursive_split produced an oversized piece: {piece[:80]!r}")
    return _pack(pieces, limit, span)


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip())]
    return [part for part in parts if part]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = math.ceil(percentile / 100.0 * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def _default_embed_fn(texts: list[str]) -> list[list[float]]:
    try:
        from ingest.embedder import embed_texts
    except ImportError as exc:
        raise ChunkingError(
            "semantic_split needs sentence embeddings; ingest.embedder is not "
            "available yet (Phase 4). Pass embed_fn=... to supply an embedder."
        ) from exc
    return embed_texts(texts)


def semantic_split(
    text: str,
    buffer_size: int | None = None,
    percentile: float | None = None,
    max_tokens: int | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[str]:
    window = config.SEMANTIC_BUFFER_SIZE if buffer_size is None else buffer_size
    cutoff = (
        config.SEMANTIC_BREAKPOINT_PERCENTILE if percentile is None else percentile
    )
    limit = config.CHUNK_SIZE if max_tokens is None else max_tokens
    sentences = _sentences(text)
    if len(sentences) < 2:
        return [text] if text.strip() else []

    embed = embed_fn if embed_fn is not None else _default_embed_fn
    vectors = embed(sentences)
    if len(vectors) != len(sentences):
        raise ChunkingError(
            f"embed_fn returned {len(vectors)} vectors for {len(sentences)} sentences"
        )

    distances = [
        1.0 - _cosine(vectors[index], vectors[min(index + window, len(vectors) - 1)])
        for index in range(len(vectors) - 1)
    ]
    threshold = _percentile(distances, cutoff)

    groups: list[str] = []
    current: list[str] = [sentences[0]]
    for index, sentence in enumerate(sentences[1:]):
        if distances[index] >= threshold:
            groups.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    groups.append(" ".join(current))

    out: list[str] = []
    for group in groups:
        if count_tokens(group) <= limit:
            out.append(group)
        else:
            out.extend(
                recursive_split(
                    group,
                    size=limit,
                    overlap=config.CHUNK_OVERLAP,
                    separators=config.CHUNK_SEPARATORS,
                )
            )
    return out


def _split_table(text: str, size: int) -> list[str]:
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 3:
        return [text] if text.strip() else []
    header, separator, rows = lines[0], lines[1], lines[2:]
    preamble = count_tokens(f"{header}\n{separator}")

    out: list[str] = []
    current: list[str] = []
    current_tokens = preamble
    for row in rows:
        row_tokens = count_tokens(row)
        if row_tokens + 1 > size - preamble:
            raise ChunkInvariantError(
                f"table row cannot fit inside the {size}-token cap: {row[:80]!r}"
            )
        if current and current_tokens + row_tokens + 1 > size:
            out.append("\n".join([header, separator, *current]))
            current = []
            current_tokens = preamble
        current.append(row)
        current_tokens += row_tokens + 1
    if current:
        out.append("\n".join([header, separator, *current]))
    return out


def _section_units(section: Section) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    prose: list[str] = []
    for block in section.blocks:
        if block.kind == "table":
            if prose:
                units.append(("text", "\n".join(prose)))
                prose = []
            units.append(("table", block.text))
        else:
            prose.append(block.text)
    if prose:
        units.append(("text", "\n".join(prose)))
    return units


def _split_unit(kind: str, text: str, strategy: str, size: int) -> list[str]:
    if kind == "table":
        return _split_table(text, size)
    if count_tokens(text) <= size:
        return [text]

    if strategy == "semantic":
        return semantic_split(text, max_tokens=size)

    pieces = recursive_split(text, size, config.CHUNK_OVERLAP, config.CHUNK_SEPARATORS)
    oversized = [piece for piece in pieces if count_tokens(piece) > size]
    if strategy == "hybrid" and oversized:
        log.warn("semantic_residual", chars=len(text), oversized=len(oversized))
        return semantic_split(text, max_tokens=size)
    return pieces


def _heading_prefix(section: Section) -> str:
    if not config.CHUNK_INCLUDE_HEADING:
        return ""
    return " > ".join(section.heading_path) or section.heading


def _chunk_text(section: Section, body: str) -> str:
    prefix = _heading_prefix(section)
    return body if not prefix else f"{prefix}\n{body}"


def _chunk_id(source_url: str, heading_path: Sequence[str], ordinal: int) -> str:
    key = source_url + "|".join(heading_path) + str(ordinal)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
def _make_chunk(document: Document, section: Section, ordinal: int, body: str) -> Chunk:
    return Chunk(
        chunk_id=_chunk_id(document.source_url, section.heading_path, ordinal),
        text=_chunk_text(section, body),
        scheme=document.scheme,
        scheme_name=document.scheme_name,
        category=document.category,
        section=section.heading,
        source_url=document.source_url,
        fetched_at=document.fetched_at,
        heading_path=list(section.heading_path),
        n_tokens=count_tokens(_chunk_text(section, body)),
    )


def _body_budget(section: Section, size: int) -> int:
    prefix = _heading_prefix(section)
    if not prefix:
        return size
    reserved = count_tokens(f"{prefix}\n")
    if reserved >= size:
        raise ChunkInvariantError(
            f"heading prefix {prefix!r} needs {reserved} tokens, cap is {size}"
        )
    return size - reserved


def chunk_by_heading(
    document: Document,
    max_tokens: int | None = None,
    strategy: str | None = None,
) -> list[Chunk]:
    size = config.CHUNK_SIZE if max_tokens is None else max_tokens
    chosen = config.CHUNK_STRATEGY if strategy is None else strategy
    chunks: list[Chunk] = []
    ordinal = 0
    for section in document.sections:
        budget = _body_budget(section, size)
        bodies: list[str] = []
        for kind, text in _section_units(section):
            bodies.extend(_split_unit(kind, text, chosen, budget))
        for body in bodies:
            chunks.append(_make_chunk(document, section, ordinal, body))
            ordinal += 1
    return chunks


def chunk_document(
    document: Document, strategy: str = config.CHUNK_STRATEGY
) -> list[Chunk]:
    if strategy not in config.CHUNK_STRATEGIES:
        raise ChunkingError(
            f"unknown strategy {strategy!r}; expected one of {list(config.CHUNK_STRATEGIES)}"
        )
    return chunk_by_heading(document, strategy=strategy)


def _assert_table_integrity(chunk: Chunk) -> None:
    run: list[str] = []
    for line in chunk.text.split("\n") + [""]:
        if _MARKDOWN_ROW.match(line):
            run.append(line)
            continue
        if len(run) > 1:
            widths = {line.count("|") for line in run}
            if len(widths) != 1:
                raise ChunkInvariantError(
                    f"chunk {chunk.chunk_id} has a split table row: {run[:2]}"
                )
        run = []


def assert_invariants(chunks: list[Chunk], max_tokens: int | None = None) -> None:
    size = config.CHUNK_SIZE if max_tokens is None else max_tokens
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.n_tokens > size:
            raise ChunkInvariantError(
                f"chunk {chunk.chunk_id} is {chunk.n_tokens} tokens, cap is {size}"
            )
        if not chunk.source_url.strip():
            raise ChunkInvariantError(f"chunk {chunk.chunk_id} has an empty source_url")
        if not chunk.section.strip():
            raise ChunkInvariantError(f"chunk {chunk.chunk_id} has an empty section")
        if "|" in chunk.scheme:
            raise ChunkInvariantError(f"chunk {chunk.chunk_id} has a malformed scheme")
        if chunk.chunk_id in seen:
            raise ChunkInvariantError(f"duplicate chunk_id {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        _assert_table_integrity(chunk)

    schemes = {chunk.scheme for chunk in chunks}
    for scheme in schemes:
        if len({chunk.source_url for chunk in chunks if chunk.scheme == scheme}) > 1:
            raise ChunkInvariantError(f"scheme {scheme} spans multiple source urls")


def chunk_documents(
    documents: list[Document], strategy: str = config.CHUNK_STRATEGY
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy))
    assert_invariants(chunks)
    return chunks


def dump_chunks(chunks: list[Chunk], path: Path | None = None) -> Path:
    target = path if path is not None else config.CHUNKS_JSONL
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    log.info("chunks", path=str(target), n=len(chunks))
    return target


def _section_tokens(section: Section) -> int:
    return count_tokens(_chunk_text(section, "\n".join(b.text for b in section.blocks)))


def analyze(documents: list[Document], strategy: str = config.CHUNK_STRATEGY) -> str:
    header = (
        f"{'scheme':<24} {'sections':>8} {'sections>cap':>12} {'avg_tokens':>10} "
        f"{'p90_tokens':>10} {'table_sections':>14} {'prose_sections':>14} {'chunks':>7}"
    )
    lines = [header, "-" * len(header)]
    totals = {"sections": 0, "over": 0, "chunks": 0, "tokens": []}
    size = config.CHUNK_SIZE

    for document in documents:
        chunks = chunk_document(document, strategy)
        tokens = [chunk.n_tokens for chunk in chunks]
        over = sum(1 for section in document.sections if _section_tokens(section) > size)
        table_sections = sum(
            1 for section in document.sections if any(b.kind == "table" for b in section.blocks)
        )
        p90 = _percentile(tokens, 90) if tokens else 0.0
        lines.append(
            f"{document.scheme:<24} {len(document.sections):>8} {over:>12} "
            f"{statistics.fmean(tokens) if tokens else 0:>10.1f} {p90:>10.1f} "
            f"{table_sections:>14} {len(document.sections) - table_sections:>14} "
            f"{len(chunks):>7}"
        )
        totals["sections"] += len(document.sections)
        totals["over"] += over
        totals["chunks"] += len(chunks)
        totals["tokens"].extend(tokens)

    tokens = totals["tokens"]
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<24} {totals['sections']:>8} {totals['over']:>12} "
        f"{statistics.fmean(tokens) if tokens else 0:>10.1f} "
        f"{_percentile(tokens, 90) if tokens else 0:>10.1f} "
        f"{'':>14} {'':>14} {totals['chunks']:>7}"
    )
    lines.append("")
    lines.append(
        f"strategy={strategy} cap={size} overlap={config.CHUNK_OVERLAP} "
        f"tokenizer={config.TOKENIZER_ENCODING} (proxy estimate, not MiniLM's)"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk the parsed documents and report the size distribution."
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="print the per-scheme chunk size table and write artifacts/chunks.jsonl",
    )
    parser.add_argument(
        "--strategy",
        default=config.CHUNK_STRATEGY,
        choices=list(config.CHUNK_STRATEGIES),
        help="override config.CHUNK_STRATEGY for this run",
    )
    args = parser.parse_args()

    if not args.analyze:
        parser.print_help()
        return 0

    try:
        documents = load_documents()
        chunks = chunk_documents(documents, args.strategy)
        dump_chunks(chunks)
        print()
        print(analyze(documents, args.strategy))
    except (ChunkingError, StructureError) as exc:
        log.error("aborted", reason=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
