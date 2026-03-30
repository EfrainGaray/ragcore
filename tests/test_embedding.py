"""Tests for ragcore.embedding — LocalEmbedder, OpenAICompatibleEmbedder, build_embedder."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------------------


def test_local_embedder_encode_returns_list():
    """LocalEmbedder.encode() wraps SentenceTransformer and delegates correctly."""
    from ragcore.embedding import LocalEmbedder

    embedder = LocalEmbedder("all-MiniLM-L6-v2")
    result = embedder.encode(["hello world"])
    # _FakeSentenceTransformer returns a numpy array; we just need it to be indexable
    assert len(result) == 1


def test_local_embedder_model_name():
    from ragcore.embedding import LocalEmbedder

    embedder = LocalEmbedder("all-MiniLM-L6-v2")
    assert embedder.model_name == "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# OpenAICompatibleEmbedder
# ---------------------------------------------------------------------------


def _fake_openai_response(texts: list[str]) -> dict:
    """Build a minimal OpenAI-compatible embeddings response."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.1 * (i + 1)] * 8}
            for i in range(len(texts))
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }


def test_openai_embedder_calls_endpoint():
    """OpenAICompatibleEmbedder posts to /v1/embeddings and parses the response."""
    from ragcore.embedding import OpenAICompatibleEmbedder

    mock_response = MagicMock()
    mock_response.json.return_value = _fake_openai_response(["hello"])
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        embedder = OpenAICompatibleEmbedder(
            model_name="text-embedding-3-small",
            api_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
        result = embedder.encode(["hello"])

    mock_post.assert_called_once()
    assert len(result) == 1
    assert len(result[0]) == 8


def test_openai_embedder_batch():
    """Multiple texts return one embedding vector per text, in order."""
    from ragcore.embedding import OpenAICompatibleEmbedder

    texts = ["hello", "world", "foo"]
    mock_response = MagicMock()
    mock_response.json.return_value = _fake_openai_response(texts)
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_response):
        embedder = OpenAICompatibleEmbedder("model", "https://api.openai.com/v1", "key")
        result = embedder.encode(texts)

    assert len(result) == 3
    # index=0 → embedding = [0.1]*8, index=1 → [0.2]*8
    assert abs(result[0][0] - 0.1) < 1e-6
    assert abs(result[1][0] - 0.2) < 1e-6


def test_openai_embedder_alias_huggingface():
    """'huggingface' alias resolves to the HF Inference API endpoint."""
    from ragcore.embedding import OpenAICompatibleEmbedder

    mock_response = MagicMock()
    mock_response.json.return_value = _fake_openai_response(["hi"])
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        embedder = OpenAICompatibleEmbedder("model", "huggingface", "hf_key")
        embedder.encode(["hi"])

    called_url = mock_post.call_args[0][0]
    assert "api-inference.huggingface.co" in called_url


def test_openai_embedder_alias_nvidia():
    """'nvidia' alias resolves to NVIDIA NIM endpoint."""
    from ragcore.embedding import OpenAICompatibleEmbedder

    mock_response = MagicMock()
    mock_response.json.return_value = _fake_openai_response(["hi"])
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        embedder = OpenAICompatibleEmbedder("model", "nvidia", "nvapi_key")
        embedder.encode(["hi"])

    called_url = mock_post.call_args[0][0]
    assert "integrate.api.nvidia.com" in called_url


# ---------------------------------------------------------------------------
# build_embedder factory
# ---------------------------------------------------------------------------


def test_build_embedder_local():
    """EMBEDDING_PROVIDER=local returns a LocalEmbedder."""
    from ragcore.config import Settings
    from ragcore.embedding import LocalEmbedder, build_embedder

    cfg = Settings(embedding_provider="local", embedding_model="all-MiniLM-L6-v2")
    embedder = build_embedder(cfg)
    assert isinstance(embedder, LocalEmbedder)
    assert embedder.model_name == "all-MiniLM-L6-v2"


def test_build_embedder_openai():
    """EMBEDDING_PROVIDER=openai returns an OpenAICompatibleEmbedder."""
    from ragcore.config import Settings
    from ragcore.embedding import OpenAICompatibleEmbedder, build_embedder

    cfg = Settings(
        embedding_provider="openai",
        embedding_api_url="https://api.openai.com/v1",
        embedding_api_key="sk-test",
        embedding_model="text-embedding-3-small",
    )
    embedder = build_embedder(cfg)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.model_name == "text-embedding-3-small"


def test_build_embedder_openai_missing_url_raises():
    """build_embedder raises ValueError when provider=openai but URL is empty."""
    from ragcore.config import Settings
    from ragcore.embedding import build_embedder

    cfg = Settings(embedding_provider="openai", embedding_api_url="", embedding_api_key="key")
    with pytest.raises(ValueError, match="EMBEDDING_API_URL"):
        build_embedder(cfg)
