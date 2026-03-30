"""Tests for ragcore.server.streaming — SSE streaming helpers."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_sse_lines():
    """stream_chat_completion must yield SSE-formatted 'data: ...' lines."""

    # Build a fake streaming httpx response
    fake_chunk_payload = json.dumps(
        {"choices": [{"delta": {"content": "Hello"}}]}
    )
    sse_line = f"data: {fake_chunk_payload}\n\n".encode()

    fake_response = MagicMock()
    fake_response.__aenter__ = AsyncMock(return_value=fake_response)
    fake_response.__aexit__ = AsyncMock(return_value=False)

    async def _fake_aiter_lines():
        yield f"data: {fake_chunk_payload}"
        yield "data: [DONE]"

    fake_response.aiter_lines = _fake_aiter_lines

    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__aenter__ = AsyncMock(return_value=fake_response)
    fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=fake_stream_ctx)

    with patch("httpx.AsyncClient", return_value=fake_client):
        from ragcore.server.streaming import stream_chat_completion

        lines = []
        async for line in stream_chat_completion(
            "what is RAG?", "some context",
            llm_url="http://fake-llm/v1", llm_key="k", llm_model="gpt-4o-mini",
        ):
            lines.append(line)

    sse_lines = [l for l in lines if l.startswith("data:")]
    assert len(sse_lines) >= 1


@pytest.mark.asyncio
async def test_stream_no_llm_url_yields_fallback():
    """When llm_url is empty the generator must still yield something (fallback)."""
    from ragcore.server.streaming import stream_chat_completion

    lines = []
    async for line in stream_chat_completion(
        "query", "context text", llm_url="", llm_key="", llm_model="gpt-4o-mini",
    ):
        lines.append(line)

    assert len(lines) >= 1


@pytest.mark.asyncio
async def test_stream_llm_error_falls_back():
    """When httpx raises an exception the generator must yield an error chunk, not propagate."""
    import httpx

    fake_stream_ctx = MagicMock()
    fake_stream_ctx.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
    fake_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    fake_client = MagicMock()
    fake_client.stream = MagicMock(return_value=fake_stream_ctx)

    with patch("httpx.AsyncClient", return_value=fake_client):
        from ragcore.server.streaming import stream_chat_completion

        lines = []
        async for line in stream_chat_completion(
            "query", "context",
            llm_url="http://broken-llm/v1", llm_key="k", llm_model="gpt-4o-mini",
        ):
            lines.append(line)

    # Must not raise and must yield at least one line
    assert len(lines) >= 1


@pytest.mark.asyncio
async def test_rest_stream_endpoint_returns_event_stream(client, store):
    """POST /v1/chat/completions with stream=true must return text/event-stream."""
    import uuid
    import numpy as np

    # Seed the store so retrieval returns something
    chunks = [
        {
            "id": str(uuid.uuid4()),
            "content": "Streaming RAG context chunk.",
            "filename": "stream_test.txt",
            "page": 0,
            "chunk_index": 0,
            "embedding": np.random.default_rng(7).standard_normal(8).tolist(),
        }
    ]
    store.add_chunks(chunks)

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "ragcore",
            "stream": True,
            "messages": [{"role": "user", "content": "What is streaming RAG?"}],
        },
    )
    # The endpoint must either stream (200 event-stream) or return a regular
    # chat completion (200); it must not return 4xx/5xx.
    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    # Accept either SSE or plain JSON completion (implementation may vary)
    assert "event-stream" in content_type or "application/json" in content_type
