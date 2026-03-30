"""Tests for ragcore.retrieval.Retriever."""

from __future__ import annotations

import pytest

from ragcore.models import SearchResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_store(store, n: int = 3) -> None:
    """Add *n* dummy chunks to *store*."""
    import uuid
    import numpy as np

    chunks = [
        {
            "id": str(uuid.uuid4()),
            "content": f"Document chunk number {i} about machine learning and AI systems.",
            "filename": "test.txt",
            "page": 0,
            "chunk_index": i,
            "embedding": np.random.default_rng(i).standard_normal(8).tolist(),
        }
        for i in range(n)
    ]
    store.add_chunks(chunks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_results(retriever, store):
    """search() must return a well-formed SearchResponse when chunks exist."""
    _seed_store(store, n=3)
    response = await retriever.search("machine learning")

    assert isinstance(response, SearchResponse)
    assert len(response.results) > 0
    for r in response.results:
        assert r.id
        assert r.content
        assert isinstance(r.score, float)
        assert r.filename == "test.txt"


@pytest.mark.asyncio
async def test_search_empty_returns_empty_list(retriever):
    """search() on an empty store must return zero results without error."""
    response = await retriever.search("anything")

    assert isinstance(response, SearchResponse)
    assert response.results == []
    assert response.total == 0


@pytest.mark.asyncio
async def test_search_applies_top_n_override(retriever, store):
    """top_n override must cap the number of returned results."""
    _seed_store(store, n=10)
    response = await retriever.search("machine learning", top_n=2)

    assert len(response.results) <= 2


@pytest.mark.asyncio
async def test_search_latency_ms_populated(retriever, store):
    """latency_ms must be a positive float."""
    _seed_store(store, n=2)
    response = await retriever.search("test query")

    assert response.latency_ms > 0
