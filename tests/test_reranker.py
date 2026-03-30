"""Tests for ragcore.reranker — LocalReranker, RemoteReranker, build_reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# LocalReranker
# ---------------------------------------------------------------------------


def test_local_reranker_returns_scores():
    from ragcore.reranker import LocalReranker

    reranker = LocalReranker("cross-encoder/ms-marco-MiniLM-L-2-v2")
    pairs = [["what is RAG?", "RAG is retrieval-augmented generation."],
             ["what is RAG?", "Python is a programming language."]]
    scores = reranker.predict(pairs)
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)


def test_local_reranker_model_name():
    from ragcore.reranker import LocalReranker

    reranker = LocalReranker("cross-encoder/ms-marco-MiniLM-L-2-v2")
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-2-v2"


def test_local_reranker_empty_pairs():
    from ragcore.reranker import LocalReranker

    reranker = LocalReranker("cross-encoder/ms-marco-MiniLM-L-2-v2")
    assert reranker.predict([]) == []


# ---------------------------------------------------------------------------
# RemoteReranker
# ---------------------------------------------------------------------------


def _fake_rerank_response(n: int) -> dict:
    """Build a minimal Cohere/Jina-style rerank response (sorted by score desc)."""
    return {
        "results": [
            {"index": i, "relevance_score": 1.0 / (i + 1)}
            for i in range(n)
        ]
    }


def test_remote_reranker_calls_endpoint():
    from ragcore.reranker import RemoteReranker

    mock_resp = MagicMock()
    mock_resp.json.return_value = _fake_rerank_response(2)
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        reranker = RemoteReranker("rerank-v3.5", "https://api.cohere.com/v1", "key")
        scores = reranker.predict([
            ["query", "doc0"],
            ["query", "doc1"],
        ])

    mock_post.assert_called_once()
    assert len(scores) == 2


def test_remote_reranker_scores_in_input_order():
    """Scores must be returned in the same order as input pairs (not API order)."""
    from ragcore.reranker import RemoteReranker

    # API returns: index=1 score=0.9, index=0 score=0.5 (sorted by relevance)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_resp):
        reranker = RemoteReranker("model", "cohere", "key")
        scores = reranker.predict([["q", "doc0"], ["q", "doc1"]])

    assert abs(scores[0] - 0.5) < 1e-6
    assert abs(scores[1] - 0.9) < 1e-6


def test_remote_reranker_empty_pairs():
    from ragcore.reranker import RemoteReranker

    with patch("httpx.Client.post") as mock_post:
        reranker = RemoteReranker("model", "cohere", "key")
        assert reranker.predict([]) == []

    mock_post.assert_not_called()


def test_remote_reranker_alias_cohere():
    from ragcore.reranker import RemoteReranker

    mock_resp = MagicMock()
    mock_resp.json.return_value = _fake_rerank_response(1)
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        reranker = RemoteReranker("rerank-v3.5", "cohere", "key")
        reranker.predict([["q", "doc"]])

    assert "cohere.com" in mock_post.call_args[0][0]


def test_remote_reranker_alias_jina():
    from ragcore.reranker import RemoteReranker

    mock_resp = MagicMock()
    mock_resp.json.return_value = _fake_rerank_response(1)
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        reranker = RemoteReranker("jina-reranker-v2-base-multilingual", "jina", "key")
        reranker.predict([["q", "doc"]])

    assert "jina.ai" in mock_post.call_args[0][0]


def test_remote_reranker_alias_voyage():
    from ragcore.reranker import RemoteReranker

    mock_resp = MagicMock()
    mock_resp.json.return_value = _fake_rerank_response(1)
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        reranker = RemoteReranker("voyage-rerank-2", "voyage", "pa-key")
        reranker.predict([["q", "doc"]])

    assert "voyageai.com" in mock_post.call_args[0][0]


# ---------------------------------------------------------------------------
# build_reranker factory
# ---------------------------------------------------------------------------


def test_build_reranker_local():
    from ragcore.config import Settings
    from ragcore.reranker import LocalReranker, build_reranker

    cfg = Settings(rerank_provider="local")
    reranker = build_reranker(cfg)
    assert isinstance(reranker, LocalReranker)


def test_build_reranker_cohere():
    from ragcore.config import Settings
    from ragcore.reranker import RemoteReranker, build_reranker

    cfg = Settings(
        rerank_provider="cohere",
        rerank_api_key="test-key",
    )
    reranker = build_reranker(cfg)
    assert isinstance(reranker, RemoteReranker)
    assert reranker.model_name == "rerank-v3.5"   # default model for cohere alias


def test_build_reranker_jina():
    from ragcore.config import Settings
    from ragcore.reranker import RemoteReranker, build_reranker

    cfg = Settings(
        rerank_provider="jina",
        rerank_api_key="jina-key",
    )
    reranker = build_reranker(cfg)
    assert isinstance(reranker, RemoteReranker)
    assert "jina" in reranker.model_name


def test_build_reranker_missing_key_raises():
    from ragcore.config import Settings
    from ragcore.reranker import build_reranker

    cfg = Settings(rerank_provider="cohere", rerank_api_key="")
    with pytest.raises(ValueError, match="RERANK_API_KEY"):
        build_reranker(cfg)


def test_build_reranker_custom_model_preserved():
    """If the user sets a custom RERANK_MODEL it should not be overridden."""
    from ragcore.config import Settings
    from ragcore.reranker import RemoteReranker, build_reranker

    cfg = Settings(
        rerank_provider="cohere",
        rerank_api_key="key",
        rerank_model="rerank-english-v3.0",
    )
    reranker = build_reranker(cfg)
    assert reranker.model_name == "rerank-english-v3.0"
