"""Tests for Query Expansion — ragcore.query_expansion.QueryExpander.

The ragcore.query_expansion module is written by the Architecture agent.
Until it exists every test is skipped gracefully via importorskip.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Guard — skip entire module if implementation isn't ready
# ---------------------------------------------------------------------------

qe_mod = pytest.importorskip(
    "ragcore.query_expansion",
    reason="ragcore.query_expansion not yet implemented (Architecture agent pending)",
)

QueryExpander = qe_mod.QueryExpander


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALT_LINES = "alternative query one\nalternative query two\nalternative query three"

_FAKE_LLM_RESPONSE_3 = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": _ALT_LINES,
            }
        }
    ]
}

_FAKE_LLM_RESPONSE_DUPLICATE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                # Second line is a repeat of the original query
                "content": "alternative one\noriginal query\nalternative two",
            }
        }
    ]
}

_FAKE_LLM_RESPONSE_2 = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "alt query A\nalt query B",
            }
        }
    ]
}


def _make_mock_response(payload: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _make_expander(**kwargs) -> QueryExpander:
    return QueryExpander(api_url="http://localhost:11434", api_key="test-key", **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_expander_returns_list_with_original():
    """expand() must always include the original query as the first element."""
    expander = _make_expander()

    with patch("httpx.Client.post", return_value=_make_mock_response(_FAKE_LLM_RESPONSE_3)):
        result = expander.expand("original query")

    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0] == "original query", (
        f"First element must be the original query, got {result[0]!r}"
    )


def test_expander_parses_llm_lines():
    """When LLM returns 'alt1\\nalt2\\nalt3', expand() must return [original, alt1, alt2, alt3]."""
    expander = _make_expander()

    with patch("httpx.Client.post", return_value=_make_mock_response(_FAKE_LLM_RESPONSE_3)):
        result = expander.expand("original query")

    assert result == [
        "original query",
        "alternative query one",
        "alternative query two",
        "alternative query three",
    ], f"Unexpected expansion result: {result}"


def test_expander_calls_llm_once():
    """A single expand() call must issue exactly one LLM POST request."""
    expander = _make_expander()

    with patch("httpx.Client.post", return_value=_make_mock_response(_FAKE_LLM_RESPONSE_3)) as mock_post:
        expander.expand("what is retrieval augmented generation?")

    assert mock_post.call_count == 1, (
        f"Expected exactly 1 LLM call, got {mock_post.call_count}"
    )


def test_expander_deduplicates():
    """If the LLM echoes the original query back, it must appear only once in the result."""
    expander = _make_expander()

    with patch("httpx.Client.post", return_value=_make_mock_response(_FAKE_LLM_RESPONSE_DUPLICATE)):
        result = expander.expand("original query")

    # "original query" must appear exactly once
    assert result.count("original query") == 1, (
        f"Duplicate original query found in results: {result}"
    )


def test_expander_n_controls_count():
    """n=2 must request 2 alternatives from the LLM (prompt should mention n)."""
    expander = _make_expander()

    with patch("httpx.Client.post", return_value=_make_mock_response(_FAKE_LLM_RESPONSE_2)) as mock_post:
        result = expander.expand("test query", n=2)

    # Verify LLM was called once
    mock_post.assert_called_once()

    # The prompt passed to the LLM should mention the number 2
    call_kwargs = mock_post.call_args.kwargs
    body = call_kwargs.get("json", {})
    messages = body.get("messages", [])
    prompt_text = " ".join(m.get("content", "") for m in messages)
    assert "2" in prompt_text, (
        f"Prompt should contain n=2 to request 2 alternatives, got: {prompt_text!r}"
    )

    # Result should contain original + up to 2 alternatives
    assert len(result) <= 3, f"n=2 should yield at most 3 items (original + 2), got {len(result)}"
    assert result[0] == "test query"
