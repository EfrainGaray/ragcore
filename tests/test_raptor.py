"""Tests for ragcore.raptor — RaptorIndexer hierarchical summarisation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from tests.conftest import _FakeSentenceTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_indexer(llm_url: str = "", levels: int = 2):
    from ragcore.raptor import RaptorIndexer

    return RaptorIndexer(
        embed_model=_FakeSentenceTransformer(),
        llm_url=llm_url,
        llm_key="",
        llm_model="gpt-4o-mini",
        levels=levels,
    )


def _make_chunks(n: int = 6) -> list[dict]:
    return [
        {
            "id": str(i),
            "content": f"This is chunk number {i} about some topic in the document.",
            "filename": "doc.txt",
            "page": 0,
            "chunk_index": i,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_raptor_build_tree_returns_summary_chunks():
    """build_tree() must return a non-empty list of summary chunks."""
    indexer = _make_indexer()
    chunks = _make_chunks(6)
    result = indexer.build_tree(chunks)
    assert isinstance(result, list)
    assert len(result) > 0


def test_raptor_summary_chunks_have_correct_type():
    """Every returned chunk must carry chunk_type == 'raptor_summary'."""
    indexer = _make_indexer()
    chunks = _make_chunks(6)
    result = indexer.build_tree(chunks)
    for chunk in result:
        assert chunk.get("chunk_type") == "raptor_summary"


def test_raptor_summary_has_raptor_level():
    """Every summary chunk must have raptor_level >= 1."""
    indexer = _make_indexer()
    chunks = _make_chunks(6)
    result = indexer.build_tree(chunks)
    for chunk in result:
        assert "raptor_level" in chunk
        assert chunk["raptor_level"] >= 1


def test_raptor_empty_input():
    """build_tree([]) must return an empty list without raising."""
    indexer = _make_indexer()
    result = indexer.build_tree([])
    assert result == []


def test_raptor_fallback_without_llm():
    """build_tree must work (concatenation fallback) even when llm_url is empty."""
    indexer = _make_indexer(llm_url="")
    chunks = _make_chunks(4)
    # Should not raise; fallback uses truncated concatenation
    result = indexer.build_tree(chunks)
    assert isinstance(result, list)
    # Content of each summary chunk should be a non-empty string
    for chunk in result:
        assert isinstance(chunk.get("content", ""), str)
        assert len(chunk["content"]) > 0
