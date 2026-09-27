from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

import config
from rag import llm
from rag.guards import check, detect_advice_intent, detect_projection, refusal_answer
from rag.logging_utils import get_logger
from rag.prompts import SENTINEL, SYSTEM_PROMPT, user_message
from rag.query_expand import detect_scheme, expand
from rag.retriever import Hit, passes_gate, retrieve
from rag.schemas import Answer, Block, Citation, QueryTrace, RefusalType

log = get_logger("chain")

LAST_UPDATED_PREFIX = "Last updated from sources:"

_TRAILING_SUFFIX = re.compile(
    r"\s*" + re.escape(LAST_UPDATED_PREFIX) + r"\s*[^\n]*$", re.IGNORECASE
)
_SENTINEL_NORMALISED = re.sub(r"[^a-z]", "", SENTINEL.lower())

_ENCODER = None


@dataclass
class Verified:
    text: str
    warnings: list[str] = field(default_factory=list)
    refusal_type: str | None = None


_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "rs.",
    "no.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "vs.",
    "etc.",
    "approx.",
    "p.a.",
    "w.e.f.",
)

_PLACEHOLDER = "\u0000"
_URL_PATTERN = re.compile(r"https?://[^\s)\]<>\"]+")
_PLACEHOLDER_PERIOD = re.compile(r"(?<=\d)\.(?=\d)")


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding(config.TOKENIZER_ENCODING)
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _render(hit: Hit) -> str:
    path = " > ".join(hit.heading_path) if hit.heading_path else hit.section
    label = hit.scheme_name or hit.scheme
    return f"[{label} | {path}]\nSource: {hit.source_url}\n\n{hit.text}"


def build_context(hits: list[Hit]) -> list[Block]:
    if not hits:
        return []

    from ingest.embedder import embed_texts

    kept = list(hits)
    dropped: list[str] = []
    if len(kept) > 1:
        vectors = np.asarray(embed_texts([hit.text for hit in kept]), dtype=np.float32)
        similarity = vectors @ vectors.T
        survivors = [0]
        for index in range(1, len(kept)):
            if any(
                float(similarity[index, prior]) > config.CONTEXT_DEDUPE_SIMILARITY
                for prior in survivors
            ):
                dropped.append(kept[index].chunk_id)
                continue
            survivors.append(index)
        kept = [kept[index] for index in survivors]

    while len(kept) > 1:
        total = sum(count_tokens(_render(hit)) for hit in kept)
        if total <= config.CONTEXT_TOKEN_BUDGET:
            break
        dropped.append(kept.pop().chunk_id)

    blocks = [Block(kind="context", text=_render(hit)) for hit in kept]
    if dropped:
        log.info("context_trimmed", dropped=len(dropped), kept=len(blocks))
    return blocks


def last_updated_from(hits: list[Hit]) -> str:
    stamps = [hit.fetched_at for hit in hits if hit.fetched_at]
    return max(stamps) if stamps else ""


def _protect(text: str) -> str:
    for abbreviation in _ABBREVIATIONS:
        stem = re.escape(abbreviation[:-1])
        text = re.sub(
            rf"\b{stem}\.",
            stem.replace("\\", "") + _PLACEHOLDER,
            text,
            flags=re.IGNORECASE,
        )
    return _PLACEHOLDER_PERIOD.sub(_PLACEHOLDER, text)


def split_sentences(text: str) -> list[str]:
    protected = _protect((text or "").strip())
    if not protected:
        return []
    parts = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [part.replace(_PLACEHOLDER, ".").strip() for part in parts]
    return [sentence for sentence in sentences if sentence]


def count_sentences(text: str) -> int:
    return len(split_sentences(text))


def verify(text: str, hits: list[Hit]) -> Verified:
    warnings: list[str] = []
    stripped = (text or "").strip()

    if not stripped:
        return Verified(
            "", ["empty model output"], RefusalType.OUT_OF_CORPUS.value
        )

    body = _TRAILING_SUFFIX.sub("", stripped).strip()
    normalised = re.sub(r"[^a-z]", "", body.lower().strip("*_`\"' "))
    if normalised == _SENTINEL_NORMALISED or normalised.startswith(_SENTINEL_NORMALISED):
        return Verified(
            "",
            ["model returned the out-of-corpus sentinel"],
            RefusalType.OUT_OF_CORPUS.value,
        )

    if body != stripped:
        warnings.append("replaced model-supplied last-updated line with the corpus date")

    sentences = split_sentences(body)
    if len(sentences) > 3:
        body = " ".join(sentences[:3]).strip()
        warnings.append(f"truncated {len(sentences)} sentences to the first 3")

    context_urls = {hit.source_url for hit in hits if hit.source_url}
    urls = _URL_PATTERN.findall(body)
    if not any(url.rstrip(".,;") in context_urls for url in urls):
        warnings.append(
            "output cited no context URL; attaching the top-1 hit source_url"
        )

    if detect_advice_intent(body):
        warnings.append("output contained advice language; replaced with static refusal")
        return Verified(body, warnings, RefusalType.ADVICE.value)
    if detect_projection(body):
        warnings.append(
            "output contained performance figures; replaced with static refusal"
        )
        return Verified(body, warnings, RefusalType.PERFORMANCE.value)

    for url in urls:
        if url.rstrip(".,;") not in context_urls:
            body = body.replace(url, "")
            warnings.append(f"stripped URL absent from context: {url}")
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r" +([.,;:!?])", r"\1", body).strip()
    return Verified(body, warnings, None)


def decorate(text: str, hits: list[Hit], trace: QueryTrace) -> Answer:
    body = _TRAILING_SUFFIX.sub("", (text or "").strip()).strip()
    last_updated = last_updated_from(hits)
    if last_updated:
        body = f"{body}\n\n{LAST_UPDATED_PREFIX} {last_updated}"
    citations: list[Citation] = []
    if hits:
        top = hits[0]
        citations.append(
            Citation(
                source_url=top.source_url,
                section=top.section,
                scheme_name=top.scheme_name,
                fetched_at=top.fetched_at,
            )
        )
    if not citations:
        trace.warnings.append("no hit available to build a citation")
    return Answer(
        text=body,
        citations=citations,
        last_updated=last_updated,
        trace=trace,
    )


def _finish(built: Answer, trace: QueryTrace, llm_called: bool) -> Answer:
    trace.guards = {**trace.guards, "llm_called": llm_called}
    built.trace = trace
    return built


def answer(question: str) -> Answer:
    trace = QueryTrace()
    question = (question or "").strip()

    started = time.perf_counter()
    guard_result = check(question)
    trace.timings_ms["guards"] = _ms(started)
    trace.guards = {
        "action": guard_result.action,
        "refusal_type": guard_result.refusal_type,
        "matched_names": list(guard_result.matched_names),
    }
    trace.warnings.extend(guard_result.warnings)
    if guard_result.action == "refuse":
        scheme = detect_scheme(question)
        return _finish(
            refusal_answer(guard_result.refusal_type, scheme), trace, llm_called=False
        )

    step = time.perf_counter()
    trace.expanded_query = expand(guard_result.query)
    trace.timings_ms["expand"] = _ms(step)

    step = time.perf_counter()
    scheme = detect_scheme(guard_result.query)
    hits = retrieve(guard_result.query, scheme)
    trace.filter = {"scheme": scheme} if scheme else None
    trace.hits = [
        {
            "chunk_id": hit.chunk_id,
            "scheme": hit.scheme,
            "section": hit.section,
            "distance": round(hit.distance, 4),
        }
        for hit in hits
    ]
    trace.timings_ms["retrieve"] = _ms(step)

    step = time.perf_counter()
    gated = passes_gate(hits)
    trace.timings_ms["gate"] = _ms(step)
    if not gated:
        trace.warnings.append("score gate rejected every hit")
        log.info("gate_failed", scheme=scheme or "none")
        return _finish(
            refusal_answer(RefusalType.OUT_OF_CORPUS.value, scheme), trace, llm_called=False
        )

    step = time.perf_counter()
    blocks = build_context(hits)
    trace.timings_ms["context"] = _ms(step)

    step = time.perf_counter()
    text = llm.generate(SYSTEM_PROMPT, user_message(blocks, question))
    trace.timings_ms["generate"] = _ms(step)

    step = time.perf_counter()
    verified = verify(text, hits)
    trace.warnings.extend(verified.warnings)
    trace.timings_ms["verify"] = _ms(step)
    if verified.refusal_type is not None:
        return _finish(
            refusal_answer(verified.refusal_type, scheme), trace, llm_called=True
        )

    step = time.perf_counter()
    decorated = decorate(verified.text, hits, trace)
    trace.timings_ms["decorate"] = _ms(step)
    return _finish(decorated, trace, llm_called=True)
