from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from rag.logging_utils import get_logger
from rag.prompts import (
    ADVICE_REFUSAL,
    EDUCATION_LINK,
    OUT_OF_CORPUS,
    PERFORMANCE_REFUSAL,
    PII_REFUSAL,
    factsheet_link,
)
from rag.schemas import Answer, QueryTrace, RefusalType

log = get_logger("guard")

ADVICE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bshould i\b",
        r"\bwhich (is|one) (is )?better\b",
        r"\bis it (good|worth|safe)\b",
        r"\brecommend",
        r"\bsuggest",
        r"\bworth buying\b",
        r"\bbuy or sell\b",
        r"\ballocat",
        r"\bportfolio for me\b",
        r"\bbest (scheme|fund|option)\b",
        r"\bcan i invest in\b",
        r"\bwhich fund should\b",
        r"\bopinion\b",
    )
)

PII_PATTERNS = {
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "aadhaar": re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?91[- ]?)?[6-9]\d{9}\b"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "account": re.compile(
        r"\b(acct|account|folio)(?:\s*(?:no\.?|num(?:ber)?|is|was|:|=))*\s*\d{6,}\b"
    ),
    "otp": re.compile(
        r"\b(otp|one time password)(?:\s*(?:is|was|:|=))*\s*\d{4,6}\b"
    ),
}

PERFORMANCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\breturns? of\b",
        r"\bcagr\b",
        r"\bbest performing\b",
        r"\bwhich gave.*return",
        r"\bperformance vs\b",
        r"\b1 year return\b",
        r"\bsince inception return\b",
        r"\bnav history\b",
    )
)

QUOTED_SPAN = re.compile(
    r"[\"'‘’“”][^\"'‘’“”]*[\"'‘’“”]"
)

NEGATION_WINDOW = re.compile(r"\b(not|n't|never|avoid)\b", re.IGNORECASE)

ANSWER = "answer"
REFUSE = "refuse"


@dataclass
class GuardResult:
    action: str
    refusal_type: str | None = None
    matched_names: list[str] = field(default_factory=list)
    query: str = ""
    query_chars: int = 0
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


def _strip_quoted(query: str) -> str:
    return QUOTED_SPAN.sub(" ", query)


def _is_negated(query: str, start: int) -> bool:
    window = query[max(0, start - 24) : start]
    return bool(NEGATION_WINDOW.search(window))


def detect_advice_intent(query: str) -> bool:
    text = _strip_quoted(query)
    for pattern in ADVICE_PATTERNS:
        for match in pattern.finditer(text):
            if not _is_negated(text, match.start()):
                return True
    return False


def detect_pii(query: str) -> list[str]:
    return [name for name, pattern in PII_PATTERNS.items() if pattern.search(query)]


def detect_performance(query: str) -> bool:
    return any(pattern.search(query) for pattern in PERFORMANCE_PATTERNS)


def check(query: str) -> GuardResult:
    raw = query or ""
    warnings: list[str] = []
    limit = config.MAX_QUERY_CHARS
    truncated = len(raw) > limit
    text = raw[:limit]
    if truncated:
        warnings.append(f"query truncated to {limit} characters")
        log.warn("query_truncated", chars=len(raw), limit=limit)

    if detect_advice_intent(text):
        log.info("advice", matched=1)
        return GuardResult(
            action=REFUSE,
            refusal_type=RefusalType.ADVICE.value,
            matched_names=["advice"],
            query="",
            query_chars=len(raw),
            truncated=truncated,
            warnings=warnings,
        )

    pii = detect_pii(text)
    if pii:
        log.info("pii", matched=",".join(pii))
        warnings.append("query discarded; not embedded or stored")
        return GuardResult(
            action=REFUSE,
            refusal_type=RefusalType.PII.value,
            matched_names=pii,
            query="",
            query_chars=len(raw),
            truncated=truncated,
            warnings=warnings,
        )

    if detect_performance(text):
        log.info("performance", matched=1)
        return GuardResult(
            action=REFUSE,
            refusal_type=RefusalType.PERFORMANCE.value,
            matched_names=["performance"],
            query="",
            query_chars=len(raw),
            truncated=truncated,
            warnings=warnings,
        )

    return GuardResult(
        action=ANSWER,
        refusal_type=None,
        matched_names=[],
        query=text,
        query_chars=len(raw),
        truncated=truncated,
        warnings=warnings,
    )


_REFUSAL_TEXT = {
    RefusalType.ADVICE.value: lambda scheme: f"{ADVICE_REFUSAL}\n\nLearn more: {EDUCATION_LINK}",
    RefusalType.PII.value: lambda scheme: PII_REFUSAL,
    RefusalType.PERFORMANCE.value: lambda scheme: (
        f"{PERFORMANCE_REFUSAL}\n\nOfficial page: {factsheet_link(scheme)}"
    ),
    RefusalType.OUT_OF_CORPUS.value: lambda scheme: (
        f"{OUT_OF_CORPUS}\n\nOfficial page: {factsheet_link(scheme)}"
    ),
}


def refusal_answer(refusal_type: str, scheme: str | None = None) -> Answer:
    if refusal_type not in _REFUSAL_TEXT:
        raise ValueError(
            f"unknown refusal type {refusal_type!r}; expected one of {sorted(_REFUSAL_TEXT)}"
        )
    trace = QueryTrace(guards={"refusal_type": refusal_type, "llm_called": False})
    return Answer(
        text=_REFUSAL_TEXT[refusal_type](scheme),
        citations=[],
        last_updated="",
        trace=trace,
        refusal=True,
        refusal_type=refusal_type,
    )
