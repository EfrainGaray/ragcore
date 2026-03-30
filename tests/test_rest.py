"""Tests for ragcore.server.rest — FastAPI + OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_store(store, n: int = 3) -> None:
    chunks = [
        {
            "id": str(uuid.uuid4()),
            "content": f"Relevant context chunk {i} about the knowledge base.",
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
async def test_health_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "checks" in body
    assert "version" in body


@pytest.mark.asyncio
async def test_search_endpoint(client, store):
    _seed_store(store)
    resp = await client.post("/search", json={"query": "knowledge base context"})
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert "total" in body
    assert "latency_ms" in body
    assert body["query"] == "knowledge base context"


@pytest.mark.asyncio
async def test_search_empty_query_rejected(client):
    """Empty query string must be rejected with 422."""
    resp = await client.post("/search", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_upload_txt_file(client):
    """Uploading a TXT file must index at least one chunk."""
    content = b"This is a test document about artificial intelligence and machine learning."
    resp = await client.post(
        "/documents/upload",
        files={"file": ("hello.txt", content, "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "hello.txt"
    assert body["chunks_indexed"] >= 1
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_list_documents(client, store):
    _seed_store(store)
    resp = await client.get("/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    # test.txt was seeded
    filenames = [d["filename"] for d in docs]
    assert "test.txt" in filenames


@pytest.mark.asyncio
async def test_delete_document(client, store):
    _seed_store(store)
    resp = await client.delete("/documents/test.txt")
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "test.txt"
    assert body["deleted"] >= 0


@pytest.mark.asyncio
async def test_v1_models(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    ids = [m["id"] for m in body["data"]]
    assert "ragcore" in ids


@pytest.mark.asyncio
async def test_v1_embeddings(client):
    resp = await client.post(
        "/v1/embeddings",
        json={"input": ["hello world", "ragcore embeddings"], "model": "ragcore"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2
    for item in body["data"]:
        assert item["object"] == "embedding"
        assert isinstance(item["embedding"], list)
        assert len(item["embedding"]) > 0


@pytest.mark.asyncio
async def test_v1_chat_completions_returns_context_not_llm_answer(client, store):
    """chat/completions must return retrieved context JSON, NOT an LLM answer."""
    _seed_store(store)
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "ragcore",
            "messages": [
                {"role": "user", "content": "Tell me about the knowledge base"},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    # Must be OpenAI-shaped
    assert body["object"] == "chat.completion"
    assert "choices" in body
    assert len(body["choices"]) == 1

    # Content must be JSON-serialised chunks, NOT a natural-language LLM answer
    content_str = body["choices"][0]["message"]["content"]
    content = json.loads(content_str)   # must parse as JSON — proves it's not prose
    assert "results" in content
    assert "query" in content
    assert "_note" in content  # our disclaimer note
    assert "LLM" in content["_note"] or "retrieved" in content["_note"]
