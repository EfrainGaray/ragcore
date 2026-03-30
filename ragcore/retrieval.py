"""Retrieval pipeline: embed → vector search → rerank → SearchResponse."""

from __future__ import annotations

import asyncio
import time
from functools import partial

from loguru import logger

from ragcore.config import Settings
from ragcore.models import SearchResponse, SearchResult
from ragcore.store.chroma import RagStore


class Retriever:
    """Combines embedding, vector search, and cross-encoder reranking."""

    def __init__(
        self,
        store: RagStore,
        embedding_model,
        rerank_model,
        settings: Settings,
    ) -> None:
        self._store = store
        self._embed_model = embedding_model
        self._rerank_model = rerank_model
        self._settings = settings

    # ------------------------------------------------------------------
    # Internal helpers (CPU-bound — run off the event loop)
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        return self._embed_model.encode([text], show_progress_bar=False)[0].tolist()

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        pairs = [[query, c["content"]] for c in candidates]
        scores = self._rerank_model.predict(pairs).tolist()
        for candidate, score in zip(candidates, scores):
            candidate["score"] = float(score)
        return sorted(candidates, key=lambda c: c["score"], reverse=True)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_n: int | None = None,
        filters: dict | None = None,
    ) -> SearchResponse:
        t0 = time.perf_counter()
        effective_top_n = top_n if top_n is not None else self._settings.top_n
        loop = asyncio.get_event_loop()

        # 1. Embed query (CPU-bound)
        query_embedding = await loop.run_in_executor(None, partial(self._embed, query))

        # 2. Vector search → top_k candidates
        candidates = await loop.run_in_executor(
            None,
            partial(
                self._store.search,
                query_embedding,
                self._settings.top_k,
                filters or {},
            ),
        )
        logger.debug("Vector search returned {} candidates for query={!r}", len(candidates), query)

        # 3. CrossEncoder rerank
        ranked = await loop.run_in_executor(
            None, partial(self._rerank, query, candidates)
        )

        # 4. Trim to top_n
        top = ranked[:effective_top_n]

        results = [
            SearchResult(
                id=r["id"],
                content=r["content"],
                score=r["score"],
                filename=r["filename"],
                page=r["page"],
                chunk_index=r["chunk_index"],
                metadata=r.get("metadata", {}),
            )
            for r in top
        ]

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "search query={!r} candidates={} results={} latency={:.1f}ms",
            query,
            len(candidates),
            len(results),
            latency_ms,
        )

        return SearchResponse(
            results=results,
            total=len(results),
            query=query,
            latency_ms=round(latency_ms, 2),
        )
