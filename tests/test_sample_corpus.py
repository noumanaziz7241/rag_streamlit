"""Tests for bundled sample corpus helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from memory_agent.demo.corpus import (
    SAMPLE_CORPUS_FILES,
    index_sample_corpus,
    list_sample_corpus_files,
    sample_corpus_is_indexed,
)
from memory_agent.config import SAMPLE_DATA_DIR


def test_sample_corpus_files_exist():
    for name in SAMPLE_CORPUS_FILES:
        assert (SAMPLE_DATA_DIR / name).is_file(), f"missing sample file: {name}"


def test_list_sample_corpus_files_returns_all():
    paths = list_sample_corpus_files()
    assert len(paths) == len(SAMPLE_CORPUS_FILES)
    assert all(path.parent == SAMPLE_DATA_DIR for path in paths)


def test_index_sample_corpus_calls_ingest():
    api = MagicMock()
    api.ingest_file_detailed.side_effect = [
        MagicMock(chunks=3, skipped=False),
        MagicMock(chunks=2, skipped=False),
        MagicMock(chunks=1, skipped=True),
        MagicMock(chunks=4, skipped=False),
    ]

    summary = index_sample_corpus(api)

    assert api.ingest_file_detailed.call_count == len(SAMPLE_CORPUS_FILES)
    assert summary.indexed_chunks == 9
    assert summary.skipped_files == 1
    assert summary.total_files == len(SAMPLE_CORPUS_FILES)


def test_sample_corpus_is_indexed_when_all_registered():
    api = MagicMock()
    api.list_documents.return_value = [
        MagicMock(source=name) for name in SAMPLE_CORPUS_FILES
    ]
    assert sample_corpus_is_indexed(api) is True


def test_sample_corpus_is_not_indexed_when_missing_files():
    api = MagicMock()
    api.list_documents.return_value = [MagicMock(source="memory_agent_overview.md")]
    assert sample_corpus_is_indexed(api) is False
