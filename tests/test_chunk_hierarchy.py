"""Tests for parent-child chunk hierarchy in ingest and retrieval.

The parent-child chunking feature is implemented by the Architecture agent.
Until the feature exists, tests that require new API surface (parent_child_chunks
setting, RagStore.get_parent_chunk, etc.) are collected but will fail with
AttributeError / ImportError — matching the expected "waiting on Architecture"
state.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_store_and_ingestor(parent_child: bool = True):
    """Return (store, ingestor) configured with parent_child_chunks=parent_child."""
    from ragcore.config import Settings
    from ragcore.store.chroma import RagStore
    from ragcore.store.ingest import Ingestor
    from tests.conftest import _FakeSentenceTransformer

    cfg = Settings(
        chroma_path="/tmp/test_chroma_hierarchy",
        chroma_collection="test_hierarchy",
        chunk_size=100,
        chunk_overlap=10,
        parent_child_chunks=parent_child,
    )
    store = RagStore(cfg)
    embed_model = _FakeSentenceTransformer()
    ingestor = Ingestor(store=store, embedding_model=embed_model, settings=cfg)
    return store, ingestor


def _sample_text(n_words: int = 200) -> bytes:
    words = ["word" + str(i % 50) for i in range(n_words)]
    return (" ".join(words)).encode()


# ---------------------------------------------------------------------------
# Parent-child ingestion tests
# ---------------------------------------------------------------------------


def test_parent_child_chunking_stores_parent_chunks():
    """When parent_child_chunks=True, the ingestor must store both child AND
    parent chunks (i.e. total stored > child-only count)."""
    store, ingestor = _make_store_and_ingestor(parent_child=True)

    count = ingestor.ingest("doc.txt", _sample_text(200))

    # There should be something stored
    assert count > 0

    # With parent-child, the store should also hold parent chunks
    # (typically stored in a separate namespace or with a type flag)
    total = store.count()
    assert total > count, (
        "Parent chunks should be stored in addition to child chunks, "
        f"but store.count()={total} <= ingest_count={count}"
    )


def test_parent_child_chunks_have_parent_id():
    """Child chunks must carry a 'parent_id' in their metadata."""
    from ragcore.config import Settings
    from ragcore.store.chroma import RagStore
    from ragcore.store.ingest import Ingestor
    from tests.conftest import _FakeSentenceTransformer

    cfg = Settings(
        chroma_path="/tmp/test_chroma_hierarchy2",
        chroma_collection="test_hierarchy2",
        chunk_size=60,
        chunk_overlap=5,
        parent_child_chunks=True,
    )
    store = RagStore(cfg)
    embed_model = _FakeSentenceTransformer()

    captured_chunks: list[dict] = []
    original_add = store.add_chunks

    def capturing_add(chunks):
        captured_chunks.extend(chunks)
        return original_add(chunks)

    store.add_chunks = capturing_add
    ingestor = Ingestor(store=store, embedding_model=embed_model, settings=cfg)
    ingestor.ingest("doc.txt", _sample_text(120))

    # Filter to child chunks only (those that should have parent_id)
    child_chunks = [c for c in captured_chunks if c.get("chunk_type") == "child"]

    assert len(child_chunks) > 0, "No child chunks found; expected some when parent_child_chunks=True"
    for chunk in child_chunks:
        assert "parent_id" in chunk, (
            f"Child chunk missing 'parent_id' in metadata: {chunk}"
        )
        assert chunk["parent_id"], "parent_id must be non-empty"


def test_parent_chunk_size_larger_than_child():
    """Parent chunks must be longer (more characters) than child chunks."""
    from ragcore.config import Settings
    from ragcore.store.chroma import RagStore
    from ragcore.store.ingest import Ingestor
    from tests.conftest import _FakeSentenceTransformer

    cfg = Settings(
        chroma_path="/tmp/test_chroma_hierarchy3",
        chroma_collection="test_hierarchy3",
        chunk_size=60,
        chunk_overlap=5,
        parent_child_chunks=True,
    )
    store = RagStore(cfg)
    embed_model = _FakeSentenceTransformer()

    captured_chunks: list[dict] = []
    original_add = store.add_chunks

    def capturing_add(chunks):
        captured_chunks.extend(chunks)
        return original_add(chunks)

    store.add_chunks = capturing_add
    ingestor = Ingestor(store=store, embedding_model=embed_model, settings=cfg)
    ingestor.ingest("doc.txt", _sample_text(200))

    parents = [c for c in captured_chunks if c.get("chunk_type") == "parent"]
    children = [c for c in captured_chunks if c.get("chunk_type") == "child"]

    assert parents and children, "Need both parent and child chunks"

    avg_parent_len = sum(len(c["content"]) for c in parents) / len(parents)
    avg_child_len = sum(len(c["content"]) for c in children) / len(children)

    assert avg_parent_len > avg_child_len, (
        f"Parent chunks (avg {avg_parent_len:.0f} chars) should be larger than "
        f"child chunks (avg {avg_child_len:.0f} chars)"
    )


# ---------------------------------------------------------------------------
# RagStore.get_parent_chunk tests
# ---------------------------------------------------------------------------


def test_get_parent_chunk(store):
    """RagStore.get_parent_chunk(parent_id) must return the parent chunk's content."""
    # Manually add a parent chunk
    parent_id = str(uuid.uuid4())
    store.add_chunks(
        [
            {
                "id": parent_id,
                "content": "This is the full parent passage with lots of context.",
                "filename": "doc.txt",
                "page": 0,
                "chunk_index": 0,
                "chunk_type": "parent",
                "embedding": np.random.default_rng(0).standard_normal(8).tolist(),
            }
        ]
    )

    result = store.get_parent_chunk(parent_id)
    assert result is not None, "get_parent_chunk must return the parent chunk"
    assert "full parent passage" in result, (
        f"Expected parent content, got: {result!r}"
    )


def test_get_parent_chunk_not_found_returns_none(store):
    """RagStore.get_parent_chunk with an unknown id must return None."""
    result = store.get_parent_chunk("non-existent-parent-id-xyz")
    assert result is None, (
        f"Expected None for unknown parent_id, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Retriever parent expansion test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_expands_to_parent_content():
    """When parent_child_chunks=True in settings, Retriever must replace
    child chunk content with parent content in the search results."""
    from ragcore.config import Settings
    from ragcore.retrieval import Retriever
    from ragcore.store.chroma import RagStore
    from tests.conftest import _FakeSentenceTransformer, _FakeCrossEncoder

    cfg = Settings(
        chroma_path="/tmp/test_chroma_retrieval_parent",
        chroma_collection="test_parent_retrieval",
        top_k=5,
        top_n=3,
        parent_child_chunks=True,
    )
    store = RagStore(cfg)
    embed_model = _FakeSentenceTransformer()
    rerank_model = _FakeCrossEncoder()

    # Seed: one parent + one child that references it
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    parent_content = "PARENT: Full context about transformer architecture in detail."
    child_content = "transformer architecture"

    store.add_chunks(
        [
            {
                "id": parent_id,
                "content": parent_content,
                "filename": "doc.txt",
                "page": 0,
                "chunk_index": 0,
                "chunk_type": "parent",
                "embedding": np.random.default_rng(1).standard_normal(8).tolist(),
            },
            {
                "id": child_id,
                "content": child_content,
                "filename": "doc.txt",
                "page": 0,
                "chunk_index": 1,
                "chunk_type": "child",
                "parent_id": parent_id,
                "embedding": np.random.default_rng(2).standard_normal(8).tolist(),
            },
        ]
    )

    retriever = Retriever(
        store=store,
        embedding_model=embed_model,
        rerank_model=rerank_model,
        settings=cfg,
    )

    response = await retriever.search("transformer architecture")

    # At least one result should have parent content
    contents = [r.content for r in response.results]
    assert any("PARENT:" in c for c in contents), (
        "Retriever should have expanded child chunk content to parent content, "
        f"but got: {contents}"
    )
