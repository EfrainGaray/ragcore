"""Unified entry point — starts both the REST API and the MCP SSE server.

Both servers share the same Retriever instance (models loaded once).

Usage:
    python -m ragcore.main
    # or via docker-compose (each service runs its own command)
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os

import uvicorn
from loguru import logger

from ragcore.config import settings


# ---------------------------------------------------------------------------
# Model / store bootstrap (run once per process)
# ---------------------------------------------------------------------------


def _build_retriever():
    """Instantiate models and return a Retriever ready to use."""
    from ragcore.embedding import build_embedder
    from ragcore.reranker import build_reranker
    from ragcore.store.chroma import RagStore
    from ragcore.retrieval import Retriever
    from ragcore.hyde import build_hyde
    from ragcore.query_expansion import build_expander

    store = RagStore(settings)
    embed_model = build_embedder(settings)
    rerank_model = build_reranker(settings)
    hyde = build_hyde(settings)
    expander = build_expander(settings)
    return Retriever(
        store=store,
        embedding_model=embed_model,
        rerank_model=rerank_model,
        settings=settings,
        hyde=hyde,
        expander=expander,
    )


def _build_ingestor(retriever):
    from ragcore.store.ingest import Ingestor

    embed_model = retriever._embed_model

    # SemanticChunker (optional — only when semantic_chunking=True)
    semantic_chunker = None
    if settings.semantic_chunking:
        try:
            from ragcore.chunking.semantic import SemanticChunker  # type: ignore
            semantic_chunker = SemanticChunker(
                embed_model=embed_model,
                threshold=settings.semantic_chunk_threshold,
                max_chunk_size=settings.semantic_chunk_max_size,
            )
        except Exception as exc:
            logger.warning("SemanticChunker unavailable: {}", exc)

    # RaptorIndexer (optional — only when raptor_enabled=True)
    raptor_indexer = None
    if settings.raptor_enabled:
        try:
            from ragcore.raptor import RaptorIndexer  # type: ignore
            raptor_indexer = RaptorIndexer(
                embed_model=embed_model,
                llm_url=settings.raptor_llm_url or settings.hyde_llm_url,
                llm_key=settings.raptor_llm_key,
                llm_model=settings.raptor_llm_model,
                levels=settings.raptor_levels,
            )
        except Exception as exc:
            logger.warning("RaptorIndexer unavailable: {}", exc)

    return Ingestor(
        store=retriever._store,
        embedding_model=embed_model,
        settings=settings,
        semantic_chunker=semantic_chunker,
        raptor_indexer=raptor_indexer,
    )


# ---------------------------------------------------------------------------
# REST process
# ---------------------------------------------------------------------------


def _run_rest() -> None:
    """Run the FastAPI server (blocking)."""
    retriever = _build_retriever()
    ingestor = _build_ingestor(retriever)

    from ragcore.server.rest import create_app

    app = create_app(retriever=retriever, ingestor=ingestor, settings=settings)
    logger.info("Starting REST API on {}:{}", settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


# ---------------------------------------------------------------------------
# MCP process
# ---------------------------------------------------------------------------


def _run_mcp() -> None:
    """Run the FastMCP SSE server (blocking)."""
    retriever = _build_retriever()

    from ragcore.server.mcp import mcp, set_retriever

    set_retriever(retriever)
    logger.info("Starting MCP SSE server on {}:{}", settings.host, settings.mcp_port)
    mcp.run(transport="sse", host=settings.host, port=settings.mcp_port)


# ---------------------------------------------------------------------------
# Main — spawn both processes
# ---------------------------------------------------------------------------


def main() -> None:
    rest_proc = multiprocessing.Process(target=_run_rest, name="ragcore-rest", daemon=True)
    mcp_proc = multiprocessing.Process(target=_run_mcp, name="ragcore-mcp", daemon=True)

    rest_proc.start()
    mcp_proc.start()

    logger.info("ragcore started — REST={} MCP={}", settings.port, settings.mcp_port)

    try:
        rest_proc.join()
        mcp_proc.join()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        rest_proc.terminate()
        mcp_proc.terminate()


if __name__ == "__main__":
    main()
