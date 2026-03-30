"""Tests for BM25 hybrid search and RRF fusion.

The ragcore.store.bm25 module (BM25Index, rrf_fuse) is written by the
Architecture agent.  Until it exists every test is skipped gracefully via an
importorskip guard so the overall suite stays green.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Guard — skip entire module if the implementation isn't ready yet
# ---------------------------------------------------------------------------

bm25_mod = pytest.importorskip(
    "ragcore.store.bm25",
    reason="ragcore.store.bm25 not yet implemented (Architecture agent pending)",
)

BM25Index = bm25_mod.BM25Index
rrf_fuse = bm25_mod.rrf_fuse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DOCS = [
    (
        "doc-1",
        "The quick brown fox jumps over the lazy dog",
    ),
    (
        "doc-2",
        "Machine learning and retrieval augmented generation systems",
    ),
    (
        "doc-3",
        "Neural networks learn representations from large corpora",
    ),
]


def _build_index() -> BM25Index:
    ids = [d[0] for d in _DOCS]
    texts = [d[1] for d in _DOCS]
    return BM25Index(ids=ids, corpus=texts)


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------


def test_bm25_build_and_search():
    """build index from 3 docs, search returns (id, score) tuples."""
    index = _build_index()
    results = index.search("fox jumps", top_k=3)

    assert isinstance(results, list)
    assert len(results) > 0
    for item in results:
        assert len(item) == 2, "Each result must be a (id, score) tuple"
        doc_id, score = item
        assert isinstance(doc_id, str)
        assert isinstance(score, (int, float))


def test_bm25_returns_results_sorted_desc():
    """Higher-scoring results must come first."""
    index = _build_index()
    results = index.search("machine learning retrieval", top_k=3)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), (
        f"Results not sorted descending: {scores}"
    )


def test_bm25_empty_corpus_returns_empty():
    """BM25Index built on an empty corpus must return [] without raising."""
    index = BM25Index(ids=[], corpus=[])
    results = index.search("anything", top_k=5)
    assert results == []


def test_bm25_query_not_in_corpus_returns_zero_scores():
    """A query with no term overlap must produce zero (or very low) scores for all docs."""
    index = _build_index()
    # Use tokens that are guaranteed not to appear in _DOCS
    results = index.search("zzz yyy xxx", top_k=3)

    # The contract: returns a list (possibly empty or all-zero scores)
    assert isinstance(results, list)
    for _, score in results:
        assert score == pytest.approx(0.0), (
            f"Expected zero score for out-of-vocabulary query, got {score}"
        )


def test_bm25_top_k_limits_results():
    """top_k=2 must return at most 2 results."""
    index = _build_index()
    results = index.search("learning neural networks", top_k=2)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# RRF fusion tests
# ---------------------------------------------------------------------------


def test_rrf_fusion():
    """Items appearing in both result lists must rank higher than items in only one.

    rrf_fuse(lists, k=60) computes score = Σ 1/(k + rank) for each item
    across all lists.
    """
    # list_a: [A, B, C]  — A best in semantic, B middle, C last
    # list_b: [B, A, D]  — B best in BM25, A middle, D not in list_a
    list_a = [("A", 0.9), ("B", 0.6), ("C", 0.3)]
    list_b = [("B", 0.8), ("A", 0.5), ("D", 0.2)]

    merged = rrf_fuse([list_a, list_b], k=60)

    assert isinstance(merged, list)
    merged_ids = [item_id for item_id, _ in merged]

    # A and B appear in both lists → should outrank C and D (single-list items)
    assert "A" in merged_ids
    assert "B" in merged_ids

    # Find positions
    pos = {item_id: idx for idx, (item_id, _) in enumerate(merged)}

    # Both A and B appear in two lists; C and D appear in only one
    # So A and B must rank above C (only in list_a) and D (only in list_b)
    assert pos["A"] < pos["C"], (
        f"A (2 lists) should outrank C (1 list). Positions: A={pos['A']}, C={pos['C']}"
    )
    assert pos["B"] < pos["D"], (
        f"B (2 lists) should outrank D (1 list). Positions: B={pos['B']}, D={pos['D']}"
    )
