from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RefusalType(str, Enum):
    ADVICE = "advice"
    PII = "pii"
    PERFORMANCE = "performance"
    OUT_OF_CORPUS = "out_of_corpus"


@dataclass
class Block:
    kind: str
    text: str


@dataclass
class Section:
    heading: str
    level: int
    heading_path: list[str] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Document:
    doc_id: str
    title: str
    scheme: str
    scheme_name: str
    category: str
    source_url: str
    fetched_at: str
    sections: list[Section] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    scheme: str
    scheme_name: str
    category: str
    section: str
    source_url: str
    fetched_at: str
    heading_path: list[str] = field(default_factory=list)
    n_tokens: int = 0
    block_types: str = ""


@dataclass
class Hit:
    chunk_id: str
    text: str
    scheme: str
    scheme_name: str
    category: str
    section: str
    source_url: str
    fetched_at: str
    distance: float


@dataclass
class Citation:
    source_url: str
    section: str
    scheme_name: str
    fetched_at: str


@dataclass
class QueryTrace:
    guards: dict = field(default_factory=dict)
    expanded_query: str = ""
    filter: dict | None = None
    hits: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    last_updated: str = ""
    trace: QueryTrace = field(default_factory=QueryTrace)
    refusal: bool = False
    refusal_type: str | None = None
