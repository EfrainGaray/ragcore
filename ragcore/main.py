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
    from sentence_transformers import SentenceTransformer, CrossEncoder

    from ragcore.store.chroma import RagStore
    from ragcore.retrieval import Retriever

    store = RagStore(settings)
    embed_model = SentenceTransformer(settings.embedding_model)
    rerank_model = CrossEncoder(settings.rerank_model)
    return Retriever(store=store, embedding_model=embed_model, rerank_model=rerank_model, settings=settings)


def _build_ingestor(retriever):
    from ragcore.store.ingest import Ingestor

    return Ingestor(
        store=retriever._store,
        embedding_model=retriever._embed_model,
        settings=settings,
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
