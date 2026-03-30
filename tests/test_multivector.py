"""Tests for ragcore.multivector — MultiVectorRetriever."""

from __future__ import annotations

import pytest
from tests.conftest import _FakeSentenceTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retriever():
    from ragcore.multivector import MultiVectorRetriever

    return MultiVectorRetriever(embed_model=_FakeSentenceTransformer(), store=None)


def _make_chunks(n: int = 4) -> list[dict]:
    return [
        {
            "id": str(i),
            "content": f"Sentence one of doc {i}. Sentence two of doc {i}. Sentence three.",
            "filename": "doc.txt",
            "page": 0,
            "chunk_index": i,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multivector_index_and_search():
    """After indexing, search() must return a non-empty list for a valid query."""
    retriever = _make_retriever()
    retriever.index(_make_chunks(4))
    results = retriever.search("sentence one", top_k=2)
    assert isinstance(results, list)
    assert len(results) > 0


def test_multivector_search_returns_list_of_dicts():
    """Each result must be a dict containing 'id', 'content', and 'score'."""
    retriever = _make_retriever()
    retriever.index(_make_chunks(3))
    results = retriever.search("doc sentence", top_k=2)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, dict)
        assert "id" in r
        assert "content" in r
        assert "score" in r


def test_multivector_empty_index():
    """search() on an empty index must return [] without raising."""
    retriever = _make_retriever()
    retriever.index([])
    results = retriever.search("anything", top_k=5)
    assert results == []


def test_multivector_scores_are_floats():
    """Every score in the results must be a Python float (or numpy float)."""
    retriever = _make_retriever()
    retriever.index(_make_chunks(3))
    results = retriever.search("sentence", top_k=3)
    for r in results:
        assert isinstance(r["score"], (float, int)) or hasattr(r["score"], "__float__")


def test_multivector_top_k_respected():
    """search() must return at most top_k results."""
    retriever = _make_retriever()
    retriever.index(_make_chunks(10))
    top_k = 3
    results = retriever.search("sentence", top_k=top_k)
    assert len(results) <= top_k
