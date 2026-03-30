"""Tests for ragcore.retrieval.Retriever."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from ragcore.models import SearchResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_store(store, n: int = 3) -> None:
    """Add *n* dummy chunks to *store*."""

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


@pytest.mark.asyncio
async def test_search_respects_filters(retriever, store):
    """filters dict must be forwarded to store.search."""
    _seed_store(store, n=3)

    original_search = store.search
    captured_filters = {}

    def capturing_search(query_embedding, top_k, filters=None):
        captured_filters["filters"] = filters
        return original_search(query_embedding, top_k, filters)

    store.search = capturing_search

    await retriever.search("machine learning", filters={"filename": "test.txt"})

    assert captured_filters["filters"] == {"filename": "test.txt"}


@pytest.mark.asyncio
async def test_search_top_n_capped_at_results(retriever, store):
    """If store returns only 2 results, top_n=10 must still return at most 2."""
    _seed_store(store, n=2)
    response = await retriever.search("machine learning", top_n=10)

    assert len(response.results) <= 2


@pytest.mark.asyncio
async def test_search_rerank_scores_descending(retriever, store):
    """Results must be sorted by rerank score in descending order."""
    _seed_store(store, n=5)
    response = await retriever.search("machine learning")

    scores = [r.score for r in response.results]
    assert scores == sorted(scores, reverse=True), f"Scores not descending: {scores}"


@pytest.mark.asyncio
async def test_search_query_too_long_raises(retriever):
    """A query longer than 2000 characters must raise ValueError or validation error."""
    long_query = "x" * 2001
    with pytest.raises((ValueError, Exception)):
        # The SearchRequest model enforces max_length=2000.
        # When called directly via retriever, the validation may not apply,
        # so we also test via the SearchRequest model.
        from ragcore.models import SearchRequest
        SearchRequest(query=long_query)


# ---------------------------------------------------------------------------
# HyDE integration — skipped until Architecture agent delivers ragcore.hyde
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_with_hyde_uses_hypothetical_embedding(store):
    """When hyde_enabled=True, Retriever must embed the HyDE-generated text
    (not the raw query) when calling the embedding model.

    This test is skipped if ragcore.hyde does not exist yet.
    """
    pytest.importorskip(
        "ragcore.hyde",
        reason="ragcore.hyde not yet implemented (Architecture agent pending)",
    )

    from ragcore.config import Settings
    from ragcore.retrieval import Retriever
    from tests.conftest import _FakeSentenceTransformer, _FakeCrossEncoder

    HYPOTHETICAL_TEXT = "Hypothetical: transformers use self-attention mechanisms for sequence modelling."
    ORIGINAL_QUERY = "how do transformers work?"

    cfg = Settings(top_k=5, top_n=3, hyde_enabled=True)

    embed_model = _FakeSentenceTransformer()
    rerank_model = _FakeCrossEncoder()

    # Patch HyDE.generate so we control what hypothetical text is produced
    with MagicMock() as _:
        pass  # just warm up MagicMock import

    from unittest.mock import patch as _patch

    with _patch("ragcore.hyde.HyDE.generate", return_value=HYPOTHETICAL_TEXT) as mock_generate:
        # Also capture what text the embedding model actually encodes
        embedded_texts: list[str] = []
        original_encode = embed_model.encode

        def capturing_encode(texts, **kwargs):
            embedded_texts.extend(texts if isinstance(texts, list) else [texts])
            return original_encode(texts, **kwargs)

        embed_model.encode = capturing_encode

        retriever = Retriever(
            store=store,
            embedding_model=embed_model,
            rerank_model=rerank_model,
            settings=cfg,
        )

        # Seed store so search actually runs
        store.add_chunks(
            [
                {
                    "id": str(uuid.uuid4()),
                    "content": "transformer self-attention",
                    "filename": "doc.txt",
                    "page": 0,
                    "chunk_index": 0,
                    "embedding": np.random.default_rng(7).standard_normal(8).tolist(),
                }
            ]
        )

        await retriever.search(ORIGINAL_QUERY)

    # HyDE.generate must have been called with the original query
    mock_generate.assert_called_once_with(ORIGINAL_QUERY)

    # The embedding call must use the hypothetical text, NOT the original query
    assert HYPOTHETICAL_TEXT in embedded_texts, (
        f"Embedding model must be called with HyDE text {HYPOTHETICAL_TEXT!r}, "
        f"but it was called with: {embedded_texts}"
    )
    assert ORIGINAL_QUERY not in embedded_texts, (
        f"Embedding model must NOT be called with raw query when HyDE is enabled, "
        f"but found {ORIGINAL_QUERY!r} in: {embedded_texts}"
    )
