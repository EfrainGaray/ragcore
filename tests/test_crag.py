"""Tests for ragcore.crag — CorrectiveRAG chunk filtering."""

from __future__ import annotations

import pytest
from tests.conftest import _FakeCrossEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crag(threshold: float = 0.5, web_search: bool = False):
    from ragcore.crag import CorrectiveRAG

    return CorrectiveRAG(
        rerank_model=_FakeCrossEncoder(),
        threshold=threshold,
        web_search=web_search,
    )


def _make_chunks(n: int) -> list[dict]:
    return [
        {"id": str(i), "content": f"chunk content number {i}", "score": 0.0}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_crag_filters_low_score_chunks():
    """Chunks whose rerank score falls below the threshold must be excluded."""
    # _FakeCrossEncoder returns 1/(i+1): chunk 0 → 1.0, chunk 1 → 0.5, chunk 2 → 0.333…
    # With threshold=0.6 only chunk 0 should survive.
    crag = _make_crag(threshold=0.6)
    chunks = _make_chunks(3)
    result = crag.filter_chunks("query", chunks)
    assert len(result) < len(chunks)


def test_crag_keeps_high_score_chunks():
    """Chunks with score >= threshold must be present in the output."""
    crag = _make_crag(threshold=0.9)
    chunks = _make_chunks(3)
    result = crag.filter_chunks("query", chunks)
    # Only chunk 0 (score=1.0) passes threshold=0.9
    assert len(result) >= 1
    # The surviving chunk must be one of the originals
    original_ids = {c["id"] for c in chunks}
    for r in result:
        assert r["id"] in original_ids


def test_crag_empty_input():
    """filter_chunks([]) must return [] without raising."""
    crag = _make_crag()
    result = crag.filter_chunks("query", [])
    assert result == []


def test_crag_all_filtered_returns_empty():
    """When all chunks score below threshold and web_search=False the result is []."""
    # threshold=2.0 is impossible; all chunks are filtered out
    crag = _make_crag(threshold=2.0, web_search=False)
    chunks = _make_chunks(4)
    result = crag.filter_chunks("query", chunks)
    assert result == []


def test_crag_threshold_zero_keeps_all():
    """threshold=0.0 means every chunk (score >= 0) must be kept."""
    crag = _make_crag(threshold=0.0)
    chunks = _make_chunks(5)
    result = crag.filter_chunks("query", chunks)
    assert len(result) == len(chunks)
