"""Tests for HyDE (Hypothetical Document Embeddings) — ragcore.hyde.HyDE.

The ragcore.hyde module is written by the Architecture agent.  Until it exists
every test is skipped gracefully via importorskip.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Guard — skip entire module if implementation isn't ready
# ---------------------------------------------------------------------------

hyde_mod = pytest.importorskip(
    "ragcore.hyde",
    reason="ragcore.hyde not yet implemented (Architecture agent pending)",
)

HyDE = hyde_mod.HyDE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_LLM_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hypothetical passage: transformers are sequence-to-sequence models.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}

_EMPTY_LLM_RESPONSE = {
    "id": "chatcmpl-empty",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop",
        }
    ],
    "usage": {},
}


def _make_mock_response(payload: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _make_hyde(api_url: str = "http://localhost:11434", **kwargs) -> HyDE:
    """Convenience factory that always returns a HyDE with a dummy API key."""
    return HyDE(api_url=api_url, api_key="test-key", **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hyde_generate_calls_llm():
    """HyDE.generate() must call httpx.Client.post (or httpx.post) with a
    chat/completions payload."""
    hyde = _make_hyde()

    with patch("httpx.Client.post", return_value=_make_mock_response(_FAKE_LLM_RESPONSE)) as mock_post:
        hyde.generate("What is a transformer?")

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args

    # URL argument — check it mentions completions
    url_arg = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("url", "")
    assert "completions" in url_arg, (
        f"Expected 'completions' in URL, got: {url_arg!r}"
    )

    # Body should contain 'messages'
    body = call_kwargs.kwargs.get("json") or call_kwargs.kwargs.get("data") or {}
    if isinstance(body, str):
        body = json.loads(body)
    assert "messages" in body, f"POST body must contain 'messages', got: {body}"


def test_hyde_returns_string():
    """HyDE.generate() must return a str."""
    hyde = _make_hyde()

    with patch("httpx.Client.post", return_value=_make_mock_response(_FAKE_LLM_RESPONSE)):
        result = hyde.generate("What is a transformer?")

    assert isinstance(result, str)
    assert len(result) > 0


def test_hyde_extract_content_from_response():
    """HyDE.generate() must parse choices[0].message.content from the LLM response."""
    expected = "Hypothetical passage: transformers are sequence-to-sequence models."
    hyde = _make_hyde()

    with patch("httpx.Client.post", return_value=_make_mock_response(_FAKE_LLM_RESPONSE)):
        result = hyde.generate("What is a transformer?")

    assert result == expected, (
        f"Expected extracted content {expected!r}, got {result!r}"
    )


def test_hyde_uses_alias_resolution():
    """Passing 'openai' as api_url alias must resolve to api.openai.com in the POST URL."""
    hyde = _make_hyde(api_url="openai")

    with patch("httpx.Client.post", return_value=_make_mock_response(_FAKE_LLM_RESPONSE)) as mock_post:
        hyde.generate("hello")

    url_arg = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url", "")
    assert "openai.com" in url_arg, (
        f"'openai' alias must resolve to api.openai.com, got URL: {url_arg!r}"
    )


def test_hyde_empty_response_returns_original_query():
    """If the LLM returns an empty string, HyDE must fall back to the original query."""
    original_query = "What is a transformer?"
    hyde = _make_hyde()

    with patch("httpx.Client.post", return_value=_make_mock_response(_EMPTY_LLM_RESPONSE)):
        result = hyde.generate(original_query)

    assert result == original_query, (
        f"Expected fallback to original query {original_query!r}, got {result!r}"
    )
