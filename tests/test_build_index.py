from __future__ import annotations

import sqlite3

import pytest

import config
from ingest.build_index import prune_orphan_segments

REGISTERED = "b898954b-f65b-4e30-a726-5c7781a80ebe"
ORPHAN = "22dd2464-704e-4c6b-80c0-4dd2d158555a"


def make_chroma_dir(tmp_path, registered: tuple[str, ...] = (REGISTERED,)):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    connection = sqlite3.connect(chroma_dir / "chroma.sqlite3")
    connection.execute("create table segments (id text, type text, scope text, collection text)")
    for segment_id in registered:
        connection.execute(
            "insert into segments values (?,?,?,?)",
            (segment_id, "urn:chroma:segment/vector/hnsw-local-persisted", "VECTOR", "c1"),
        )
    connection.commit()
    connection.close()
    return chroma_dir


def make_segment(chroma_dir, name: str, size: int = 1024) -> None:
    segment = chroma_dir / name
    segment.mkdir()
    (segment / "data_level0.bin").write_bytes(b"\x00" * size)
    (segment / "header.bin").write_bytes(b"\x00" * 8)


def test_prune_removes_only_unregistered_segment_dirs(tmp_path, monkeypatch) -> None:
    chroma_dir = make_chroma_dir(tmp_path)
    make_segment(chroma_dir, REGISTERED)
    make_segment(chroma_dir, ORPHAN)
    monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)

    removed = prune_orphan_segments()

    assert removed == [f"{ORPHAN}:1032"]
    assert not (chroma_dir / ORPHAN).exists()
    assert (chroma_dir / REGISTERED / "data_level0.bin").is_file()


def test_prune_keeps_every_registered_segment(tmp_path, monkeypatch) -> None:
    chroma_dir = make_chroma_dir(tmp_path, registered=(REGISTERED, ORPHAN))
    make_segment(chroma_dir, REGISTERED)
    make_segment(chroma_dir, ORPHAN)
    monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)

    assert prune_orphan_segments() == []
    assert (chroma_dir / REGISTERED).is_dir()
    assert (chroma_dir / ORPHAN).is_dir()


def test_prune_leaves_non_uuid_entries_untouched(tmp_path, monkeypatch) -> None:
    chroma_dir = make_chroma_dir(tmp_path)
    make_segment(chroma_dir, REGISTERED)
    make_segment(chroma_dir, ORPHAN)
    logs = chroma_dir / "chroma.log"
    logs.write_text("log line", encoding="utf-8")
    monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)

    prune_orphan_segments()

    assert logs.is_file()
    assert (chroma_dir / "chroma.sqlite3").is_file()
    assert not (chroma_dir / ORPHAN).exists()


def test_prune_is_noop_without_sqlite_registry(tmp_path, monkeypatch) -> None:
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    make_segment(chroma_dir, ORPHAN)
    monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)

    assert prune_orphan_segments() == []
    assert (chroma_dir / ORPHAN).is_dir()


def test_prune_is_noop_for_missing_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "absent")
    assert prune_orphan_segments() == []


def test_prune_reports_nothing_when_registry_lists_no_orphans(tmp_path, monkeypatch) -> None:
    chroma_dir = make_chroma_dir(tmp_path)
    make_segment(chroma_dir, REGISTERED)
    monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)
    assert prune_orphan_segments() == []


@pytest.mark.parametrize("name", ["not-a-uuid", "12345", ""])
def test_prune_ignores_directories_that_are_not_uuids(tmp_path, monkeypatch, name: str) -> None:
    chroma_dir = make_chroma_dir(tmp_path)
    if name:
        make_segment(chroma_dir, name)
    monkeypatch.setattr(config, "CHROMA_DIR", chroma_dir)

    assert prune_orphan_segments() == []
    if name:
        assert (chroma_dir / name).is_dir()
