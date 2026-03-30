"""Tests for ragcore.config.Settings defaults and environment overrides."""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Default value tests
# ---------------------------------------------------------------------------


def test_default_top_k():
    """settings.top_k must default to 10."""
    from ragcore.config import Settings

    cfg = Settings()
    assert cfg.top_k == 10


def test_default_top_n():
    """settings.top_n must default to 5."""
    from ragcore.config import Settings

    cfg = Settings()
    assert cfg.top_n == 5


def test_default_chunk_size():
    """settings.chunk_size must default to 512."""
    from ragcore.config import Settings

    cfg = Settings()
    assert cfg.chunk_size == 512


def test_default_embedding_model():
    """settings.embedding_model must default to 'all-MiniLM-L6-v2'."""
    from ragcore.config import Settings

    cfg = Settings()
    assert cfg.embedding_model == "all-MiniLM-L6-v2"


def test_override_via_env(monkeypatch):
    """Monkeypatching TOP_K=20 must result in settings.top_k == 20 on a fresh instance."""
    monkeypatch.setenv("TOP_K", "20")

    # Import Settings freshly — pydantic-settings reads env at instantiation time
    from ragcore.config import Settings

    cfg = Settings()
    assert cfg.top_k == 20


def test_chroma_path_default():
    """settings.chroma_path must default to './data/chroma'."""
    from ragcore.config import Settings

    cfg = Settings()
    assert cfg.chroma_path == "./data/chroma"
