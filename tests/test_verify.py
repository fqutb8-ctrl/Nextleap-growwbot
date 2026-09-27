from __future__ import annotations

import pytest

import config
import rag.llm as llm
from rag.chain import answer, count_sentences, split_sentences, verify
from rag.schemas import Hit, RefusalType

URL = config.SCHEME_URLS["hdfc_large_cap"]


def make_hit(
    distance: float = 0.20,
    section: str = "Expense ratio",
    scheme: str = "hdfc_large_cap",
    text: str = "Expense ratio: 0.69% - 0.76%",
    fetched_at: str = "2026-09-27T00:00:00Z",
) -> Hit:
    return Hit(
        chunk_id="a" * 12,
        text=text,
        scheme=scheme,
        scheme_name="HDFC Large Cap Fund - Direct Growth",
        category="large_cap",
        section=section,
        source_url=config.SCHEME_URLS[scheme],
        fetched_at=fetched_at,
        distance=distance,
        heading_path=[section],
    )


def stub(text: str):
    return lambda system, user: text


def run(monkeypatch, text: str, question: str = "What is the expense ratio of HDFC Large Cap Fund?"):
    monkeypatch.setattr(llm, "generate", stub(text))
    return answer(question)


def test_five_sentences_truncated_to_three(monkeypatch) -> None:
    result = run(
        monkeypatch,
        "One fact. Second fact. Third fact. Fourth fact. Fifth fact.",
    )
    assert result.refusal is False
    assert count_sentences(result.text.split("Last updated from sources:")[0]) == 3
    assert any("truncated 5 sentences" in w for w in result.trace.warnings)


def test_three_sentences_are_left_alone(monkeypatch) -> None:
    result = run(monkeypatch, "One fact. Second fact. Third fact.")
    assert not any("truncated" in w for w in result.trace.warnings)


def test_invented_url_is_stripped(monkeypatch) -> None:
    result = run(
        monkeypatch,
        f"The expense ratio is 0.69%. See https://example.com/fake-fund for details. {URL}",
    )
    assert "example.com" not in result.text
    assert any("stripped URL absent from context" in w for w in result.trace.warnings)


def test_advice_language_triggers_static_refusal(monkeypatch) -> None:
    result = run(monkeypatch, "I recommend you invest in this scheme for better returns.")
    assert result.refusal is True
    assert result.refusal_type == RefusalType.ADVICE.value
    assert "I recommend" not in result.text
    assert any("advice language" in w for w in result.trace.warnings)


def test_performance_language_triggers_static_refusal(monkeypatch) -> None:
    result = run(monkeypatch, "This fund is expected to return 14.2% over three years.")
    assert result.refusal is True
    assert result.refusal_type == RefusalType.PERFORMANCE.value
    assert "14.2" not in result.text


def test_output_without_url_still_gets_exactly_one_citation(monkeypatch) -> None:
    result = run(monkeypatch, "The expense ratio is between 0.69% and 0.76%.")
    assert result.refusal is False
    assert len(result.citations) == 1
    assert result.citations[0].source_url == URL
    assert any("cited no context URL" in w for w in result.trace.warnings)


def test_empty_output_is_out_of_corpus(monkeypatch) -> None:
    result = run(monkeypatch, "   ")
    assert result.refusal is True
    assert result.refusal_type == RefusalType.OUT_OF_CORPUS.value
    assert any("empty model output" in w for w in result.trace.warnings)


def test_sentinel_output_is_out_of_corpus(monkeypatch) -> None:
    result = run(monkeypatch, "I couldn't find that in my sources.")
    assert result.refusal_type == RefusalType.OUT_OF_CORPUS.value
    assert any("sentinel" in w for w in result.trace.warnings)


def test_sentinel_plus_last_updated_line_is_out_of_corpus(monkeypatch) -> None:
    result = run(
        monkeypatch,
        "I couldn't find that in my sources.\n\nLast updated from sources: 2026-09-27T06:47:02Z",
    )
    assert result.refusal is True
    assert result.refusal_type == RefusalType.OUT_OF_CORPUS.value
    assert result.citations == []


def test_sentinel_with_markdown_and_official_page_is_out_of_corpus(monkeypatch) -> None:
    result = run(
        monkeypatch,
        "**I couldn't find that in my sources.** Please check the official scheme page.\n\n"
        "Official page: https://www.hdfcmf.com/",
    )
    assert result.refusal_type == RefusalType.OUT_OF_CORPUS.value


def test_sentinel_lookalike_factual_answer_is_not_out_of_corpus(monkeypatch) -> None:
    result = run(
        monkeypatch,
        "The corpus does not list a CEO, but the fund house section names HDFC Mutual Fund.",
    )
    assert result.refusal_type is None


def test_historical_return_facts_are_answered_not_refused(monkeypatch) -> None:
    for text in [
        "The 3 year return of HDFC Large Cap Fund is 12.4%.",
        "The annualised return of the ELSS fund is 14.2% over 3 years.",
        "HDFC Large Cap has a CAGR of 13.1% since inception.",
    ]:
        result = run(monkeypatch, text)
        assert result.refusal_type is None, f"wrongly refused {text!r}"


def test_forward_looking_projections_are_refused(monkeypatch) -> None:
    for text in [
        "The fund is expected to return 15% next year.",
        "This scheme will return around 18% annually.",
        "You should return to this fund after 5 years.",
        "Returns are guaranteed at 12%.",
    ]:
        result = run(monkeypatch, text)
        assert result.refusal_type == RefusalType.PERFORMANCE.value, f"allowed {text!r}"


def test_every_bad_stub_produces_a_warning(monkeypatch) -> None:
    bad_outputs = [
        "One. Two. Three. Four. Five.",
        "Fact. See https://example.com/x",
        "I recommend this fund.",
        "A fact with no URL at all.",
        "",
    ]
    for text in bad_outputs:
        result = run(monkeypatch, text)
        assert result.trace.warnings, f"no warning for {text!r}"
        assert result.refusal or result.text


def test_model_last_updated_line_is_replaced(monkeypatch) -> None:
    result = run(
        monkeypatch,
        "The expense ratio is 0.69%. Last updated from sources: 1999-01-01",
    )
    assert "1999-01-01" not in result.text
    assert result.text.count("Last updated from sources:") == 1
    assert result.last_updated in result.text
    assert any("last-updated" in w for w in result.trace.warnings)


def test_citation_is_always_exactly_one_for_answered(monkeypatch) -> None:
    for text in [
        "A fact. Another fact.",
        f"A fact with the real link. {URL}",
        "One. Two. Three. Four. Five.",
    ]:
        result = run(monkeypatch, text)
        if not result.refusal:
            assert len(result.citations) == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("e.g. this is one sentence.", 1),
        ("i.e. one. two. three.", 3),
        ("Rs. 100 is the minimum. SIP is monthly.", 2),
        ("The TER is 0.69% to 0.76%. That is the range.", 2),
        ("No. 1 is fine. No. 2 is not.", 2),
        ("One. Two. Three. Four. Five.", 5),
        ("Casino. The next sentence.", 2),
        ("Trailing text without a period", 1),
        ("", 0),
    ],
)
def test_count_sentences(text: str, expected: int) -> None:
    assert count_sentences(text) == expected


def test_truncation_never_leaves_a_dangling_fragment() -> None:
    text = "First complete sentence here. Second complete sentence here. Third here. Dangling frag"
    sentences = split_sentences(text)
    assert len(sentences) == 4
    trimmed = " ".join(sentences[:3])
    assert trimmed.endswith(".")
    assert not trimmed.split()[-1] in ("and", "the", "of", "a")


def test_verify_passes_through_clean_output_unchanged() -> None:
    hits = [make_hit()]
    clean = f"The expense ratio is 0.69% to 0.76%. Source: {URL}"
    result = verify(clean, hits)
    assert result.refusal_type is None
    assert result.text == clean
    assert result.warnings == []


def test_verify_without_hits_still_returns_text() -> None:
    result = verify("A plain fact.", [])
    assert result.refusal_type is None
    assert result.text == "A plain fact."
    assert any("cited no context URL" in w for w in result.warnings)


def test_refusal_replacement_keeps_warning_but_not_bad_text(monkeypatch) -> None:
    result = run(monkeypatch, "You should buy this fund, it is the best scheme for you.")
    assert result.refusal is True
    assert "best scheme" not in result.text
    assert result.trace.warnings
