from __future__ import annotations

import re

import config
from rag.logging_utils import get_logger

log = get_logger("query")

SYNONYMS = config.QUERY_SYNONYMS

_WHITESPACE = re.compile(r"\s+")

_SYNONYM_PATTERNS = tuple(
    (re.compile(r"\b" + re.escape(canonical) + r"\b", re.IGNORECASE), canonical, variants)
    for canonical, variants in SYNONYMS.items()
)

_ALIAS_PATTERNS = tuple(
    (re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE), slug)
    for slug, aliases in config.SCHEME_ALIASES.items()
    for alias in aliases
)


def normalize(query: str) -> str:
    return _WHITESPACE.sub(" ", query or "").strip().lower()


def expand(query: str) -> str:
    text = normalize(query)
    if not text:
        return text

    cap = int(len(text) * config.EXPANSION_RATIO)
    out = text
    forced_used = False
    for pattern, canonical, variants in _SYNONYM_PATTERNS:
        if not pattern.search(text):
            continue
        for variant in variants:
            if variant in out.split():
                continue
            candidate = f"{out} {variant}"
            if len(candidate) <= cap:
                out = candidate
                continue
            if forced_used:
                log.warn("expansion_capped", term=canonical, cap=cap)
                break
            forced_used = True
            log.warn("expansion_over_cap", term=canonical, cap=cap)
            out = candidate

    if out != text:
        log.info("expanded", original_chars=len(text), expanded_chars=len(out))
    return out


def detect_scheme(query: str) -> str | None:
    text = normalize(query)
    matched = {slug for pattern, slug in _ALIAS_PATTERNS if pattern.search(text)}
    if len(matched) != 1:
        log.info("scheme_filter", matched_count=len(matched))
        return None
    return matched.pop()
