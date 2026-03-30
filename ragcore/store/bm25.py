"""BM25 index wrapper for ragcore hybrid search."""

from __future__ import annotations

from loguru import logger


class BM25Index:
    """In-memory BM25 index backed by rank_bm25.BM25Okapi."""

    def __init__(self, ids: list[str] | None = None, corpus: list[str] | None = None) -> None:
        self._index = None       # BM25Okapi instance or None when empty
        self._ids: list[str] = []
        if ids is not None and corpus is not None:
            self.build(corpus, ids)

    # ------------------------------------------------------------------
    # Build / rebuild
    # ------------------------------------------------------------------

    def build(self, corpus: list[str], ids: list[str]) -> None:
        """Build (or rebuild) the index from a list of texts and their IDs."""
        if not corpus:
            self._index = None
            self._ids = []
            logger.debug("BM25Index built with empty corpus")
            return

        from rank_bm25 import BM25Okapi

        tokenized = [text.lower().split() for text in corpus]
        self._index = BM25Okapi(tokenized)
        self._ids = list(ids)
        logger.debug("BM25Index built with {} documents", len(corpus))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to *top_k* (id, bm25_score) pairs sorted descending.

        Returns an empty list when the index is empty.
        """
        if self._index is None or not self._ids:
            return []

        tokenized_query = query.lower().split()
        raw_scores = self._index.get_scores(tokenized_query)
        scores: list[float] = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

        # Pair with ids and sort by score descending
        pairs = sorted(zip(self._ids, scores), key=lambda x: x[1], reverse=True)
        return pairs[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion — module-level helper
# ---------------------------------------------------------------------------


def rrf_fuse(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Each list is a sequence of (id, score) pairs sorted by descending score.
    Returns a merged list of (id, rrf_score) pairs sorted by rrf_score descending.
    Items appearing in more lists rank higher.

    Formula: rrf_score(d) = Σ_list  1 / (k + rank(d, list))
    """
    rrf_scores: dict[str, float] = {}
    for result_list in result_lists:
        for rank, (doc_id, _score) in enumerate(result_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
