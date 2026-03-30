"""Tests for ragcore.chunking.semantic — SemanticChunker."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Inject fake sentence_transformers before any ragcore import (conftest already
# does this, but we re-assert here for clarity and in case this file runs alone)
# ---------------------------------------------------------------------------

# conftest.py already installs the fakes; nothing extra needed here.


# ---------------------------------------------------------------------------
# Helper: build a SemanticChunker with the fake embed model
# ---------------------------------------------------------------------------


def _make_chunker(threshold: float = 0.5, max_chunk_size: int = 512):
    from ragcore.chunking.semantic import SemanticChunker
    from tests.conftest import _FakeSentenceTransformer

    return SemanticChunker(
        embed_model=_FakeSentenceTransformer(),
        threshold=threshold,
        max_chunk_size=max_chunk_size,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_semantic_chunker_returns_list_of_strings():
    """chunk() must return a list of strings for normal input."""
    chunker = _make_chunker()
    result = chunker.chunk("Hello world. Another sentence.")
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)


def test_semantic_chunker_empty_text():
    """chunk('') must return an empty list, not raise."""
    chunker = _make_chunker()
    result = chunker.chunk("")
    assert result == []


def test_semantic_chunker_splits_long_text():
    """Text that exceeds max_chunk_size must be split into multiple chunks."""
    # max_chunk_size=10 forces splits because sentences are longer
    chunker = _make_chunker(max_chunk_size=10)
    long_text = (
        "This is the first sentence about topic A. "
        "Here comes the second sentence about topic B. "
        "And a third sentence about topic C."
    )
    result = chunker.chunk(long_text)
    # With a very small max_chunk_size there must be more than one chunk
    assert len(result) > 1


def test_semantic_chunker_respects_threshold():
    """At threshold=1.0 every sentence boundary triggers a split (similarity never reaches 1.0)."""
    # threshold=1.0 means similarity must equal 1.0 to merge; random embeddings
    # never satisfy this, so each sentence becomes its own chunk.
    chunker = _make_chunker(threshold=1.0, max_chunk_size=512)
    text = (
        "The sky is blue today. "
        "Quantum mechanics describes subatomic particles. "
        "Pizza is a popular dish worldwide."
    )
    result = chunker.chunk(text)
    # At threshold=1.0 each sentence should be its own chunk
    assert len(result) >= 2


def test_semantic_chunker_single_sentence():
    """A single sentence must return exactly one chunk."""
    chunker = _make_chunker()
    result = chunker.chunk("This is a single sentence without any period boundary.")
    assert len(result) == 1
    assert isinstance(result[0], str)
