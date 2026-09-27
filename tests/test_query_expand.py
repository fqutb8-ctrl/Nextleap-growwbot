from __future__ import annotations

import pytest

import config
from rag.query_expand import SYNONYMS, detect_scheme, expand, normalize


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  Exit   LOAD  ", "exit load"),
        ("minimum\tsip amount", "minimum sip amount"),
        ("", ""),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    "query,scheme",
    [
        ("elss lock in period", "hdfc_elss"),
        ("tax saver benefit", "hdfc_elss"),
        ("what is the large cap fund benchmark", "hdfc_large_cap"),
        ("largecap fund", "hdfc_large_cap"),
        ("flexi cap fund", "hdfc_flexi_cap"),
        ("hdfc equity fund direct growth", "hdfc_flexi_cap"),
        ("small cap fund", "hdfc_small_cap"),
        ("balanced advantage fund", "hdfc_balanced_advantage"),
    ],
)
def test_detect_scheme_single_hit(query: str, scheme: str) -> None:
    assert detect_scheme(query) == scheme


def test_detect_scheme_none_on_no_match() -> None:
    assert detect_scheme("what is the exit load") is None


def test_detect_scheme_none_on_ambiguity() -> None:
    assert detect_scheme("compare large cap and small cap") is None


def test_detect_scheme_three_schemes_is_none() -> None:
    assert detect_scheme("elss versus large cap versus small cap") is None


def test_two_aliases_for_same_scheme_are_not_ambiguous() -> None:
    assert detect_scheme("elss tax saver exit load") == "hdfc_elss"


def test_expand_appends_synonyms() -> None:
    assert "exit charge" in expand("exit load")
    assert "systematic investment plan" in expand("minimum sip amount")
    assert "holding period" in expand("what is the lock in")


def test_expand_guarantees_one_variant_even_when_over_cap() -> None:
    result = expand("exit load")
    assert result.startswith("exit load")
    assert "exit charge" in result


def test_expand_caps_long_queries() -> None:
    query = "exit load and ter and sip and riskometer and direct plan for my portfolio"
    text = normalize(query)
    assert len(expand(query)) <= int(len(text) * config.EXPANSION_RATIO) + 32


def test_expand_uses_word_boundaries_not_substrings() -> None:
    result = expand("exit load after maturity for water and materials")
    assert "expense ratio" not in result
    assert "total expense ratio" not in result


def test_expand_leaves_clean_query_untouched() -> None:
    assert expand("who is the fund manager") == "who is the fund manager"


def test_expand_preserves_normalized_original() -> None:
    assert expand("  EXIT   Load ").startswith("exit load")


def test_expand_of_empty_query() -> None:
    assert expand("") == ""


def test_synonyms_come_from_config() -> None:
    assert SYNONYMS is config.QUERY_SYNONYMS
    assert set(SYNONYMS) == {
        "sip",
        "lock in",
        "ter",
        "exit load",
        "riskometer",
        "direct plan",
    }
