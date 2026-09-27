from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api
import config
import rag.llm as llm
from rag.llm import LLMError
from rag.prompts import DISCLAIMER
from rag.retriever import IndexMissingError
from rag.schemas import Answer, Citation, QueryTrace

client = TestClient(api.app, raise_server_exceptions=False)

PAN = "ABCDE1234F"

ANSWER_TEXT = (
    "HDFC Small Cap Fund - Direct Growth charges an exit load of 1% if units are "
    "redeemed within 1 year. Last updated from sources: 2026-09-27T06:47:02Z"
)
FETCHED_AT = "2026-09-27T06:47:02Z"


def grounded_answer() -> Answer:
    trace = QueryTrace(
        guards={"action": "answer", "refusal_type": None, "matched_names": []},
        expanded_query="exit load",
        filter={"scheme": "hdfc_small_cap"},
        hits=[
            {
                "chunk_id": "4153d3e985da",
                "scheme": "hdfc_small_cap",
                "section": "Exit load",
                "distance": 0.31,
            }
        ],
        timings_ms={"retrieve": 12.5, "generate": 420.0},
    )
    return Answer(
        text=ANSWER_TEXT,
        citations=[
            Citation(
                source_url=config.SCHEME_URLS["hdfc_small_cap"],
                section="Exit load",
                scheme_name="HDFC Small Cap Fund - Direct Growth",
                fetched_at=FETCHED_AT,
            )
        ],
        last_updated=FETCHED_AT,
        trace=trace,
    )


def explode(*args, **kwargs):
    raise AssertionError("generate() must not be called on this path")


def test_health_reports_index_state() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in ("ok", "degraded")
    assert isinstance(payload["index_ready"], bool)
    assert payload["collection"] == config.COLLECTION_NAME
    assert payload["sources"] == len(config.SCHEME_URLS)


def test_sources_lists_every_allowlisted_page() -> None:
    response = client.get("/sources")
    assert response.status_code == 200
    rows = response.json()
    assert {row["scheme"] for row in rows} == set(config.SCHEME_URLS)
    assert all(row["url"].startswith("https://groww.in/") for row in rows)


def test_ask_rejects_empty_question() -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_ask_rejects_oversized_question() -> None:
    response = client.post("/ask", json={"question": "a" * (config.MAX_QUERY_CHARS + 1)})
    assert response.status_code == 422


def test_advice_refusal_never_calls_llm(monkeypatch) -> None:
    monkeypatch.setattr(llm, "generate", explode)
    response = client.post("/ask", json={"question": "Should I buy HDFC Large Cap?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["refusal"] is True
    assert payload["refusal_type"] == "advice"
    assert payload["citations"] == []
    assert payload["trace"]["guards"]["llm_called"] is False


def test_performance_refusal_never_calls_llm(monkeypatch) -> None:
    monkeypatch.setattr(llm, "generate", explode)
    response = client.post(
        "/ask", json={"question": "What are the 5 year returns of HDFC Flexi Cap?"}
    )
    assert response.status_code == 200
    assert response.json()["refusal_type"] == "performance"


def test_pii_is_refused_and_never_echoed(monkeypatch) -> None:
    monkeypatch.setattr(llm, "generate", explode)
    response = client.post("/ask", json={"question": f"my pan is {PAN}"})
    assert response.status_code == 200
    body = response.text
    assert PAN not in body
    payload = response.json()
    assert payload["refusal_type"] == "pii"
    assert payload["question"] == api.REDACTED_QUESTION
    assert payload["trace"]["guards"]["llm_called"] is False


def test_missing_index_returns_503(monkeypatch) -> None:
    def missing(question: str):
        raise IndexMissingError("collection not found")

    monkeypatch.setattr(api, "answer", missing)
    response = client.post("/ask", json={"question": "What is the expense ratio?"})
    assert response.status_code == 503
    assert response.json()["detail"] == api.INDEX_UNAVAILABLE


def test_llm_failure_returns_503(monkeypatch) -> None:
    def failed(question: str):
        raise LLMError("boom")

    monkeypatch.setattr(api, "answer", failed)
    response = client.post("/ask", json={"question": "What is the expense ratio?"})
    assert response.status_code == 503
    assert response.json()["detail"] == api.LLM_UNAVAILABLE


def test_unexpected_failure_returns_500(monkeypatch) -> None:
    def broken(question: str):
        raise ValueError("boom")

    monkeypatch.setattr(api, "answer", broken)
    response = client.post("/ask", json={"question": "What is the expense ratio?"})
    assert response.status_code == 500
    assert response.json()["detail"] == api.ANSWER_FAILED


def test_successful_answer_carries_citation_and_trace(monkeypatch) -> None:
    monkeypatch.setattr(api, "answer", lambda question: grounded_answer())
    response = client.post(
        "/ask", json={"question": "What is the exit load on HDFC Small Cap?"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "What is the exit load on HDFC Small Cap?"
    assert payload["answer"] == ANSWER_TEXT
    assert payload["refusal"] is False
    assert payload["refusal_type"] is None
    assert payload["disclaimer"] == DISCLAIMER
    assert payload["last_updated"] == FETCHED_AT
    assert payload["citations"] == [
        {
            "source_url": config.SCHEME_URLS["hdfc_small_cap"],
            "section": "Exit load",
            "scheme_name": "HDFC Small Cap Fund - Direct Growth",
            "fetched_at": FETCHED_AT,
        }
    ]
    assert payload["trace"]["filter"] == {"scheme": "hdfc_small_cap"}
    assert payload["trace"]["hits"][0]["chunk_id"] == "4153d3e985da"


def test_answer_without_citation_is_still_serialised(monkeypatch) -> None:
    def bare(question: str) -> Answer:
        result = grounded_answer()
        result.citations = []
        return result

    monkeypatch.setattr(api, "answer", bare)
    response = client.post("/ask", json={"question": "What is the exit load?"})
    assert response.status_code == 200
    assert response.json()["citations"] == []


def test_include_trace_false_omits_trace(monkeypatch) -> None:
    monkeypatch.setattr(api, "answer", lambda question: grounded_answer())
    response = client.post(
        "/ask",
        json={"question": "What is the exit load?", "include_trace": False},
    )
    assert response.status_code == 200
    assert response.json()["trace"] is None


def test_root_redirects_to_docs() -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 302)
    assert response.headers["location"] == "/docs"


def test_openapi_schema_builds() -> None:
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "/ask" in schema.json()["paths"]
