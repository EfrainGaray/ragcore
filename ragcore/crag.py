"""Corrective RAG — post-retrieval relevance filtering."""
from __future__ import annotations
from loguru import logger


class CorrectiveRAG:
    """Filters retrieved chunks by relevance score, optionally supplements with web search.

    Uses the cross-encoder rerank model to score (query, chunk) pairs.
    Chunks below the threshold are discarded.
    """

    def __init__(self, rerank_model, threshold: float = 0.5, web_search: bool = False) -> None:
        self._model = rerank_model
        self._threshold = threshold
        self._web_search = web_search

    def filter_chunks(self, query: str, chunks: list[dict]) -> list[dict]:
        """Score chunks with cross-encoder, return only those above threshold.

        chunks: list of dicts with at least "content" key (same format as retrieval results)
        Returns: filtered list, preserving original order and scores
        """
        if not chunks:
            return []

        pairs = [[query, c["content"]] for c in chunks]
        raw = self._model.predict(pairs)
        scores = raw.tolist() if hasattr(raw, "tolist") else list(raw)

        filtered = []
        for chunk, score in zip(chunks, scores):
            if float(score) >= self._threshold:
                c = dict(chunk)
                c["crag_score"] = float(score)
                filtered.append(c)

        logger.debug(
            "CRAG: {}/{} chunks passed threshold={}",
            len(filtered), len(chunks), self._threshold
        )

        if not filtered and self._web_search:
            logger.info("CRAG: all chunks filtered, web search supplement not yet implemented")
            # Future: call web search API and return results

        return filtered


def build_crag(settings, rerank_model) -> CorrectiveRAG | None:
    """Factory — returns CorrectiveRAG if crag_enabled, else None."""
    if not getattr(settings, "crag_enabled", False):
        return None
    return CorrectiveRAG(
        rerank_model=rerank_model,
        threshold=getattr(settings, "crag_threshold", 0.5),
        web_search=getattr(settings, "crag_web_search", False),
    )
