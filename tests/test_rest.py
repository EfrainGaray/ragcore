"""Tests for ragcore.server.rest — FastAPI + OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import sys
import types
import uuid
from unittest.mock import MagicMock, patch

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


@pytest.mark.asyncio
async def test_search_with_top_n_override(client, store):
    """POST /search with top_n=2 must return at most 2 results."""
    _seed_store(store, n=10)
    resp = await client.post("/search", json={"query": "knowledge base context", "top_n": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) <= 2


@pytest.mark.asyncio
async def test_search_empty_query_returns_422(client):
    """Empty string query must return 422 Unprocessable Entity."""
    resp = await client.post("/search", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_upload_pdf_file(client):
    """Uploading fake PDF bytes (with mocked ingest) must return 200."""
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "PDF content about knowledge base retrieval."

    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = MagicMock(return_value=fake_reader)

    with patch.dict(sys.modules, {"pypdf": fake_pypdf}):
        resp = await client.post(
            "/documents/upload",
            files={"file": ("doc.pdf", b"%PDF-fake-content", "application/pdf")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "doc.pdf"
    assert body["chunks_indexed"] >= 1
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_upload_unsupported_format(client):
    """Uploading an unsupported .zip file must return 400 or 422."""
    resp = await client.post(
        "/documents/upload",
        files={"file": ("archive.zip", b"PK\x03\x04fake-zip-content", "application/zip")},
    )
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_delete_nonexistent_file_returns_zero(client):
    """DELETE /documents/ghost.txt for a non-existent file must return {"deleted": 0}."""
    resp = await client.delete("/documents/ghost.txt")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == 0


@pytest.mark.asyncio
async def test_health_degraded_when_chroma_fails(retriever, ingestor):
    """When the store raises an exception, the health endpoint should surface an error."""
    from httpx import AsyncClient
    from httpx._transports.asgi import ASGITransport
    from ragcore.server.rest import create_app

    # Patch the store's count() to simulate a failure
    original_count = retriever._store.count
    retriever._store.count = MagicMock(side_effect=RuntimeError("chroma unavailable"))

    app = create_app(retriever=retriever, ingestor=ingestor)
    degraded_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    try:
        # The health endpoint does not catch store exceptions, so we expect
        # either an HTTP 500 response or the exception to propagate.
        try:
            resp = await degraded_client.get("/health")
            # If a response was returned it must be a server error or degraded
            assert resp.status_code in (500, 503) or (
                resp.status_code == 200
                and resp.json().get("status") in ("degraded", "error")
            ), f"Expected error response, got {resp.status_code}: {resp.text}"
        except RuntimeError as exc:
            # The exception propagated through the ASGI transport — this also
            # demonstrates that chroma failure affects health
            assert "chroma unavailable" in str(exc)
    finally:
        retriever._store.count = original_count
        await degraded_client.aclose()


@pytest.mark.asyncio
async def test_v1_chat_completions_extracts_last_user_message(client, store):
    """With 3 messages, the query must be extracted from the last user message."""
    _seed_store(store, n=3)
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "ragcore",
            "messages": [
                {"role": "user", "content": "First question about something"},
                {"role": "assistant", "content": "Some answer"},
                {"role": "user", "content": "Final question about knowledge base"},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    content_str = body["choices"][0]["message"]["content"]
    content = json.loads(content_str)
    assert content["query"] == "Final question about knowledge base"


@pytest.mark.asyncio
async def test_v1_chat_completions_empty_messages_rejected(client):
    """An empty messages list must return 422 Unprocessable Entity."""
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "ragcore", "messages": []},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_v1_embeddings_batch(client):
    """Sending a list of 3 strings must return exactly 3 embedding vectors."""
    resp = await client.post(
        "/v1/embeddings",
        json={
            "input": ["first text", "second text", "third text"],
            "model": "ragcore",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 3
    for item in body["data"]:
        assert item["object"] == "embedding"
        assert isinstance(item["embedding"], list)
        assert len(item["embedding"]) > 0
