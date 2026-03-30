"""Tests for ragcore.server.mcp — FastMCP tool functions."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

import ragcore.server.mcp as mcp_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_store(store, n: int = 3) -> None:
    chunks = [
        {
            "id": str(uuid.uuid4()),
            "content": f"MCP test chunk {i} with useful knowledge.",
            "filename": "mcp_doc.txt",
            "page": 0,
            "chunk_index": i,
            "embedding": np.random.default_rng(i + 10).standard_normal(8).tolist(),
        }
        for i in range(n)
    ]
    store.add_chunks(chunks)


# ---------------------------------------------------------------------------
# Tests
#
# We test the tool *functions* directly (not the MCP protocol) by injecting
# a Retriever via mcp_module.set_retriever().
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_knowledge_base_tool(retriever, store):
    _seed_store(store)
    mcp_module.set_retriever(retriever)

    result = await mcp_module.search_knowledge_base("useful knowledge")

    assert isinstance(result, dict)
    assert "results" in result
    assert "query" in result
    assert "total" in result
    assert "latency_ms" in result


@pytest.mark.asyncio
async def test_list_documents_tool(retriever, store):
    _seed_store(store)
    mcp_module.set_retriever(retriever)

    result = await mcp_module.list_documents()

    assert isinstance(result, dict)
    assert "documents" in result
    assert "count" in result
    assert result["count"] >= 1
    filenames = [d["filename"] for d in result["documents"]]
    assert "mcp_doc.txt" in filenames


@pytest.mark.asyncio
async def test_get_document_count_tool(retriever, store):
    _seed_store(store)
    mcp_module.set_retriever(retriever)

    result = await mcp_module.get_document_count()

    assert isinstance(result, dict)
    assert "total_chunks" in result
    assert "total_documents" in result
    assert result["total_chunks"] >= 3
    assert result["total_documents"] >= 1


@pytest.mark.asyncio
async def test_search_tool_returns_results_list(retriever, store):
    """search_knowledge_base tool must return a dict with a 'results' key that is a list."""
    _seed_store(store)
    mcp_module.set_retriever(retriever)

    result = await mcp_module.search_knowledge_base("useful knowledge")

    assert "results" in result
    assert isinstance(result["results"], list)


@pytest.mark.asyncio
async def test_search_tool_empty_query_handled(retriever, store):
    """search_knowledge_base with an empty query must not raise and must return a dict."""
    mcp_module.set_retriever(retriever)

    # Empty store, empty query — should return gracefully
    result = await mcp_module.search_knowledge_base("")

    assert isinstance(result, dict)
    # Either has results key (possibly empty) or an error key
    assert "results" in result or "error" in result


@pytest.mark.asyncio
async def test_list_documents_tool_empty(retriever):
    """list_documents tool on an empty store must return an empty documents list."""
    mcp_module.set_retriever(retriever)

    result = await mcp_module.list_documents()

    assert isinstance(result, dict)
    assert "documents" in result
    assert isinstance(result["documents"], list)
    assert result["documents"] == []
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_get_document_count_returns_zero_when_empty(retriever):
    """get_document_count on an empty store must return count=0."""
    mcp_module.set_retriever(retriever)

    result = await mcp_module.get_document_count()

    assert isinstance(result, dict)
    assert "total_chunks" in result
    assert result["total_chunks"] == 0
    assert result["total_documents"] == 0
