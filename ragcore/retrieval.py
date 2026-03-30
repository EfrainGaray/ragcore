"""Retrieval pipeline: embed → vector search → rerank → SearchResponse."""

from __future__ import annotations

import asyncio
import time
from functools import partial

from loguru import logger

from ragcore.config import Settings
from ragcore.models import SearchResponse, SearchResult
from ragcore.store.cache import SearchCache
from ragcore.store.chroma import RagStore


class Retriever:
    """Combines embedding, vector search, and cross-encoder reranking."""

    def __init__(
        self,
        store: RagStore,
        embedding_model,
        rerank_model,
        settings: Settings,
        cache: SearchCache | None = None,
        hyde=None,
        expander=None,
    ) -> None:
        self._store = store
        self._embed_model = embedding_model
        self._rerank_model = rerank_model
        self._settings = settings
        self._cache: SearchCache = cache if cache is not None else SearchCache()

        # Auto-create HyDE/Expander from settings if not injected
        if hyde is None and settings.hyde_enabled:
            from ragcore.hyde import HyDE
            hyde = HyDE(api_url=settings.hyde_llm_url, api_key=settings.hyde_llm_key,
                        model=settings.hyde_llm_model)

        if expander is None and settings.query_expansion_enabled and settings.hyde_llm_url:
            from ragcore.query_expansion import QueryExpander
            expander = QueryExpander(api_url=settings.hyde_llm_url, api_key=settings.hyde_llm_key,
                                     model=settings.hyde_llm_model, n=settings.query_expansion_count)

        self._hyde = hyde
        self._expander = expander

    # ------------------------------------------------------------------
    # Internal helpers (CPU-bound — run off the event loop)
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        result = self._embed_model.encode([text], show_progress_bar=False)
        row = result[0]
        return row.tolist() if hasattr(row, "tolist") else list(row)

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        pairs = [[query, c["content"]] for c in candidates]
        raw = self._rerank_model.predict(pairs)
        scores: list[float] = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        for candidate, score in zip(candidates, scores):
            candidate["score"] = float(score)
        return sorted(candidates, key=lambda c: c["score"], reverse=True)

    def _vector_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int,
        filters: dict,
    ) -> list[dict]:
        """Run either hybrid or dense-only search based on settings."""
        if self._settings.hybrid_search:
            return self._store.search_hybrid(
                query_embedding, query_text, top_k, filters
            )
        return self._store.search(query_embedding, top_k, filters)

    @staticmethod
    def _rrf_fuse(result_lists: list[list[dict]], top_k: int, k: int = 60) -> list[dict]:
        """Fuse multiple result lists using Reciprocal Rank Fusion."""
        rrf_scores: dict[str, float] = {}
        best_hit: dict[str, dict] = {}

        for result_list in result_lists:
            for rank, hit in enumerate(result_list, start=1):
                doc_id = hit["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
                if doc_id not in best_hit:
                    best_hit[doc_id] = hit

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        merged = []
        for doc_id in sorted_ids[:top_k]:
            hit = dict(best_hit[doc_id])
            hit["score"] = rrf_scores[doc_id]
            merged.append(hit)
        return merged

    def _apply_parent_child(self, candidates: list[dict]) -> list[dict]:
        """Replace child content with parent content if parent_child_chunks is enabled."""
        if not self._settings.parent_child_chunks:
            return candidates

        result = []
        for hit in candidates:
            parent_id = hit.get("metadata", {}).get("parent_id")
            if parent_id:
                parent_content = self._store.get_parent_chunk(parent_id)
                if parent_content:
                    hit = dict(hit)
                    hit["content"] = parent_content
            result.append(hit)
        return result

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
        resolved_filters = filters or {}

        # 0. Cache lookup
        cache_key = SearchCache.make_key(query, top_n, resolved_filters)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for query={!r}", query)
            return cached.model_copy(update={"cache_hit": True})

        loop = asyncio.get_event_loop()

        # 1. Embed query (CPU-bound)
        # If HyDE is enabled, generate a hypothetical document and embed that instead
        if self._settings.hyde_enabled and self._hyde is not None:
            hypo = await loop.run_in_executor(None, self._hyde.generate, query)
            embed_text = hypo
            logger.debug("HyDE generated hypothetical document for query={!r}", query)
        else:
            embed_text = query

        query_embedding = await loop.run_in_executor(None, partial(self._embed, embed_text))

        # 2. Vector search (with optional query expansion / RRF fusion)
        if self._settings.query_expansion_enabled and self._expander is not None:
            # Expand query into multiple phrasings
            expanded: list[str] = await loop.run_in_executor(
                None, self._expander.expand, query
            )
            logger.debug("Query expansion produced {} variants for query={!r}", len(expanded), query)

            # Run full embed + search for each variant
            result_lists: list[list[dict]] = []
            for variant in expanded:
                if variant == query and embed_text == query:
                    # Reuse already-computed embedding for the original query
                    emb = query_embedding
                else:
                    emb = await loop.run_in_executor(None, partial(self._embed, variant))
                hits = await loop.run_in_executor(
                    None,
                    partial(
                        self._vector_search,
                        emb,
                        variant,         # BM25 uses the variant text
                        self._settings.top_k,
                        resolved_filters,
                    ),
                )
                result_lists.append(hits)

            candidates = self._rrf_fuse(result_lists, self._settings.top_k)
        else:
            candidates = await loop.run_in_executor(
                None,
                partial(
                    self._vector_search,
                    query_embedding,
                    query,
                    self._settings.top_k,
                    resolved_filters,
                ),
            )

        logger.debug("Search returned {} candidates for query={!r}", len(candidates), query)

        # 3. Parent-child chunk replacement
        candidates = await loop.run_in_executor(
            None, partial(self._apply_parent_child, candidates)
        )

        # 4. CrossEncoder rerank (always uses original query)
        ranked = await loop.run_in_executor(
            None, partial(self._rerank, query, candidates)
        )

        # 5. Trim to top_n
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

        response = SearchResponse(
            results=results,
            total=len(results),
            query=query,
            latency_ms=round(latency_ms, 2),
            cache_hit=False,
        )

        # 6. Store in cache
        self._cache.set(cache_key, response)

        return response
