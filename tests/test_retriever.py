from __future__ import annotations

import subprocess
import sys

import pytest

import config
from ingest.embedder import ModelMismatchError
from rag.retriever import (
    EmptyCollectionError,
    Hit,
    MissingCollectionError,
    RetrievalError,
    format_probe,
    get_collection,
    passes_gate,
    retrieve,
)

ALL_SCHEMES = set(config.SCHEME_URLS)

KNOWN_ANSWERS = [
    ("what is the exit load on HDFC large cap direct growth", "hdfc_large_cap"),
    ("lock in period for ELSS", "hdfc_elss"),
    ("expense ratio of HDFC small cap fund", "hdfc_small_cap"),
    ("riskometer level of balanced advantage fund", "hdfc_balanced_advantage"),
    ("who is the fund manager of flexi cap fund", "hdfc_flexi_cap"),
    ("minimum investment amount for ELSS tax saver", "hdfc_elss"),
    ("exit load for small cap fund", "hdfc_small_cap"),
    ("benchmark of large cap fund", "hdfc_large_cap"),
    ("AUM and fund house of HDFC large cap", "hdfc_large_cap"),
    ("holdings of HDFC large cap", "hdfc_large_cap"),
]


def _index_available() -> bool:
    try:
        get_collection()
    except (RetrievalError, ModelMismatchError, OSError):
        return False
    return True


requires_index = pytest.mark.skipif(
    not _index_available(),
    reason="vector index not built; run python -m ingest.build_index",
)


def make_hit(distance: float, scheme: str = "hdfc_large_cap", section: str = "Exit load") -> Hit:
    return Hit(
        chunk_id="c" * 12,
        text="Exit load of 1% if redeemed within 1 year",
        section=section,
        scheme=scheme,
        source_url=config.SCHEME_URLS.get(scheme, config.HDFC_MF_HOME),
        fetched_at="2026-09-27T00:00:00Z",
        distance=distance,
    )


def test_gate_is_false_for_no_hits() -> None:
    assert passes_gate([]) is False


def test_gate_passes_when_all_hits_within_limit() -> None:
    assert passes_gate([make_hit(0.10), make_hit(0.60)]) is True


def test_gate_fails_when_any_hit_over_limit() -> None:
    assert passes_gate([make_hit(0.10), make_hit(0.65)]) is False


def test_gate_boundary_is_inclusive() -> None:
    assert passes_gate([make_hit(config.MAX_DISTANCE)]) is True
    assert passes_gate([make_hit(config.MAX_DISTANCE + 0.001)]) is False


def test_gate_aggregation_min_uses_best_evidence(monkeypatch) -> None:
    monkeypatch.setattr(config, "GATE_AGGREGATION", "min")
    assert passes_gate([make_hit(0.10), make_hit(0.90)]) is True
    assert passes_gate([make_hit(0.70)]) is False


def test_gate_rejects_unknown_aggregation(monkeypatch) -> None:
    monkeypatch.setattr(config, "GATE_AGGREGATION", "median")
    with pytest.raises(RetrievalError):
        passes_gate([make_hit(0.10)])


def test_unknown_scheme_filter_raises() -> None:
    with pytest.raises(RetrievalError):
        retrieve("what is the exit load", scheme="not_a_scheme")


@requires_index
@pytest.mark.parametrize("question,expected", KNOWN_ANSWERS)
def test_expected_scheme_ranks_first_for_known_answer(question: str, expected: str) -> None:
    hits = retrieve(question)
    assert len(hits) == config.TOP_K
    assert hits[0].scheme == expected
    assert expected in {hit.scheme for hit in hits}


@requires_index
def test_expected_scheme_page_appears_in_top_k(question: str = "exit load") -> None:
    hits = retrieve(question, scheme="hdfc_large_cap")
    assert config.SCHEME_URLS["hdfc_large_cap"] in {hit.source_url for hit in hits}


@requires_index
def test_topk_ordering_is_stable_within_a_process() -> None:
    first = retrieve("minimum SIP amount", auto_detect=False)
    second = retrieve("minimum SIP amount", auto_detect=False)
    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]
    assert [hit.distance for hit in first] == sorted(hit.distance for hit in first)


@requires_index
@pytest.mark.xfail(
    reason="ranks 4-5 sit in a five-way tie at distance 0.4802, and Chroma resolves "
    "ties differently across processes, so top-k membership at tied distances is not "
    "reproducible. Bare 'equity fund' is also absent from config.SCHEME_ALIASES (only "
    "'hdfc equity fund'), so no filter applies. Same weakness as Phase 4 holding rows.",
)
def test_sector_allocation_question_known_gap() -> None:
    hits = retrieve("sector allocation of equity fund", auto_detect=False)
    assert "hdfc_flexi_cap" in {hit.scheme for hit in hits}


@requires_index
def test_scheme_filter_ranks_expected_scheme_first() -> None:
    hits = retrieve("what is the exit load", scheme="hdfc_large_cap")
    assert {hit.scheme for hit in hits} == {"hdfc_large_cap"}
    assert "exit load" in hits[0].section.lower() or "exit load" in hits[1].section.lower()


@requires_index
def test_multi_scheme_question_hits_multiple_schemes_without_filter() -> None:
    hits = retrieve("minimum SIP amount", auto_detect=False)
    assert len({hit.scheme for hit in hits}) >= 3

@requires_index
def test_ambiguous_question_applies_no_filter() -> None:
    hits = retrieve("compare large cap and small cap")
    assert len({hit.scheme for hit in hits}) > 1


@requires_index
def test_single_scheme_question_applies_filter() -> None:
    hits = retrieve("what is the exit load on HDFC large cap direct growth")
    assert {hit.scheme for hit in hits} == {"hdfc_large_cap"}


@requires_index
def test_hits_are_ordered_by_increasing_distance() -> None:
    hits = retrieve("what is the expense ratio", auto_detect=False)
    distances = [hit.distance for hit in hits]
    assert distances == sorted(distances)


@requires_index
def test_hits_carry_required_metadata() -> None:
    hit = retrieve("what is the exit load", scheme="hdfc_large_cap")[0]
    assert hit.chunk_id
    assert hit.text
    assert hit.section
    assert hit.scheme == "hdfc_large_cap"
    assert hit.source_url == config.SCHEME_URLS["hdfc_large_cap"]
    assert hit.fetched_at
    assert hit.distance > 0


@requires_index
def test_missing_collection_is_distinct_error(monkeypatch) -> None:
    monkeypatch.setattr(config, "COLLECTION_NAME", "collection_that_does_not_exist")
    with pytest.raises(MissingCollectionError):
        get_collection()


def test_empty_collection_is_distinct_error(tmp_path, monkeypatch) -> None:
    import chromadb

    from ingest.embedder import collection_metadata

    db = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(db))
    client.get_or_create_collection(
        name="empty_probe", metadata=collection_metadata(), embedding_function=None
    )
    monkeypatch.setattr(config, "CHROMA_DIR", db)
    monkeypatch.setattr(config, "COLLECTION_NAME", "empty_probe")
    with pytest.raises(EmptyCollectionError):
        get_collection()


def test_missing_and_empty_are_different_exception_types() -> None:
    assert not issubclass(MissingCollectionError, EmptyCollectionError)
    assert not issubclass(EmptyCollectionError, MissingCollectionError)
    assert issubclass(MissingCollectionError, RetrievalError)
    assert issubclass(EmptyCollectionError, RetrievalError)


@requires_index
def test_model_mismatch_is_surfaced(monkeypatch) -> None:
    monkeypatch.setattr(config, "EMBED_MODEL", "sentence-transformers/other-model")
    with pytest.raises(ModelMismatchError):
        get_collection()


@requires_index
def test_window_mismatch_is_surfaced(monkeypatch) -> None:
    monkeypatch.setattr(config, "EMBED_MAX_SEQ_LENGTH", 256)
    with pytest.raises(ModelMismatchError):
        get_collection()


def test_format_probe_handles_no_hits() -> None:
    output = format_probe("anything", [], None)
    assert "no hits" in output
    assert "gate: FAIL" in output


@requires_index
def test_format_probe_lists_ranked_hits() -> None:
    hits = retrieve("what is the exit load", scheme="hdfc_large_cap")
    output = format_probe("what is the exit load", hits, "hdfc_large_cap")
    assert "scheme filter: hdfc_large_cap" in output
    assert "hdfc-large-cap-fund-direct-growth" in output
    assert output.index("1. d=") < output.index("2. d=")


@requires_index
def test_probe_cli_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rag.retriever", "--probe", "minimum SIP amount"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0
    assert "question: minimum SIP amount" in result.stdout
    assert "1. d=" in result.stdout
