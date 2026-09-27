from __future__ import annotations

import csv

import pytest

import config
from rag.guards import (
    GuardResult,
    check,
    detect_advice_intent,
    detect_performance,
    detect_pii,
    refusal_answer,
)
from rag.prompts import ADVICE_REFUSAL, DISCLAIMER, EDUCATION_LINK, factsheet_link
from rag.schemas import RefusalType

ADVICE_QUERIES = [
    "should I invest in HDFC Large Cap",
    "which is better, large cap or small cap",
    "is it good to start an ELSS fund",
    "what do you recommend for a 5 year horizon",
    "Which HDFC fund gives the best returns?",
    "which scheme has the highest return",
    "best returns of HDFC large cap",
]

PII_QUERIES = [
    ("my pan is ABCDE1234F", ["pan"]),
    ("email me at ravi.sharma@example.com instead", ["email"]),
]

PERFORMANCE_QUERIES = [
    "what is the CAGR of HDFC Large Cap",
    "1 year return for the small cap fund",
    "NAV history please",
]

CLEAN_QUERIES = [
    "what is the exit load",
    "what is the expense ratio of HDFC Large Cap Fund",
    "minimum sip amount",
    "who is the fund manager",
    "what is the benchmark",
    "what is the riskometer level of HDFC ELSS Tax Saver Fund",
]


@pytest.mark.parametrize("query", ADVICE_QUERIES)
def test_advice_refuses(query: str) -> None:
    result = check(query)
    assert result.action == "refuse"
    assert result.refusal_type == RefusalType.ADVICE.value
    assert detect_advice_intent(query)


@pytest.mark.parametrize("query,expected", PII_QUERIES)
def test_pii_refuses_with_names_only(query: str, expected: list[str]) -> None:
    result = check(query)
    assert result.action == "refuse"
    assert result.refusal_type == RefusalType.PII.value
    assert detect_pii(query) == expected
    assert set(result.matched_names) == set(expected)


@pytest.mark.parametrize("query", PERFORMANCE_QUERIES)
def test_performance_refuses(query: str) -> None:
    result = check(query)
    assert result.action == "refuse"
    assert result.refusal_type == RefusalType.PERFORMANCE.value
    assert detect_performance(query)


@pytest.mark.parametrize("query", CLEAN_QUERIES)
def test_clean_queries_pass_through(query: str) -> None:
    result = check(query)
    assert result.action == "answer"
    assert result.refusal_type is None
    assert result.matched_names == []
    assert result.query == query


def test_pii_value_never_appears_in_result() -> None:
    secret = "ABCDE1234F"
    result = check("my pan is ABCDE1234F")
    assert secret in "my pan is ABCDE1234F"
    assert secret not in str(result)
    assert secret not in str(result.matched_names)
    assert secret not in str(result.query)
    assert result.query == ""


def test_pii_value_never_appears_in_logs(capsys) -> None:
    secret = "9876543210"
    check("call me on 9876543210")
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "phone" in captured.out


def test_pii_answer_text_has_no_secret() -> None:
    answer = refusal_answer(RefusalType.PII.value)
    assert answer.refusal is True
    assert answer.refusal_type == RefusalType.PII.value
    assert "9876543210" not in str(answer)


def test_advice_uses_verbatim_prd_text_and_education_link() -> None:
    answer = refusal_answer(RefusalType.ADVICE.value)
    assert ADVICE_REFUSAL in answer.text
    assert EDUCATION_LINK in answer.text
    assert DISCLAIMER not in answer.text


def test_performance_refusal_uses_factsheet_link() -> None:
    answer = refusal_answer(RefusalType.PERFORMANCE.value, scheme="hdfc_elss")
    assert factsheet_link("hdfc_elss") in answer.text
    assert "hdfc-elss-tax-saver" in answer.text


def test_refusal_answer_never_marks_llm_used() -> None:
    for refusal_type in RefusalType:
        answer = refusal_answer(refusal_type.value)
        assert answer.refusal is True
        assert answer.trace.guards["llm_called"] is False
        assert answer.citations == []


def test_unknown_refusal_type_raises() -> None:
    with pytest.raises(ValueError):
        refusal_answer("not_a_real_refusal")


def test_quoted_advice_verb_is_not_flagged() -> None:
    assert not detect_advice_intent('what does "should I invest" mean in this glossary')
    assert not detect_advice_intent("the page says 'recommend' for balanced funds")


def test_negated_advice_is_not_flagged() -> None:
    assert not detect_advice_intent("I do not think I should i buy this")


def test_pii_falls_back_to_false_positive_rather_than_leaking() -> None:
    result = check("my folio number is 12345678")
    assert result.refusal_type == RefusalType.PII.value
    assert "folio" in result.matched_names or "account" in result.matched_names
    assert "12345678" not in str(result)


def test_query_length_cap_truncates_and_warns() -> None:
    long_query = "expense ratio " * 1000
    result = check(long_query)
    assert result.truncated is True
    assert len(result.query) == config.MAX_QUERY_CHARS
    assert any("truncated" in warning for warning in result.warnings)
    assert result.query_chars == len(long_query)


def test_short_query_is_not_flagged_as_truncated() -> None:
    result = check("exit load")
    assert result.truncated is False
    assert result.warnings == []


def test_guard_result_is_a_dataclass_with_no_hidden_query() -> None:
    result = GuardResult(action="answer", query="hello")
    assert "hello" in str(result)
    assert check("hello").action == "answer"


def test_factsheet_link_falls_back_for_unknown_scheme() -> None:
    assert factsheet_link("not_a_scheme") == config.HDFC_MF_HOME
    assert factsheet_link(None) == config.HDFC_MF_HOME


def test_scheme_urls_match_sources_csv() -> None:
    with config.SOURCES_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    from_csv = {row["scheme"]: row["url"] for row in rows}
    assert config.SCHEME_URLS == from_csv
