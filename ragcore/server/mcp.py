"""FastMCP server exposing ragcore tools.

Run standalone:
    python -m ragcore.server.mcp

Or import *mcp* and mount/run programmatically from main.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import FastMCP
from loguru import logger

from ragcore.config import settings

if TYPE_CHECKING:
    from ragcore.retrieval import Retriever

# ---------------------------------------------------------------------------
# Global MCP app — tools are registered at import time via decorators.
# The *retriever* is injected at startup via set_retriever().
# ---------------------------------------------------------------------------

mcp = FastMCP("ragcore")

_retriever: "Retriever | None" = None


def set_retriever(retriever: "Retriever") -> None:
    """Inject the shared Retriever instance before serving."""
    global _retriever
    _retriever = retriever
    logger.info("MCP: retriever injected")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_knowledge_base(query: str, top_n: int = 5) -> dict:
    """Search the knowledge base for relevant context.

    Returns ranked document chunks.  Use this whenever you need background
    information from the indexed documents before composing a response.
    """
    if _retriever is None:
        return {"error": "Retriever not initialised"}

    response = await _retriever.search(query=query, top_n=top_n)
    return {
        "query": response.query,
        "total": response.total,
        "latency_ms": response.latency_ms,
        "results": [r.model_dump() for r in response.results],
    }


@mcp.tool()
async def list_documents() -> dict:
    """List all documents currently indexed in the knowledge base."""
    if _retriever is None:
        return {"error": "Retriever not initialised"}

    docs = _retriever._store.list_documents()
    return {
        "documents": [d.model_dump() for d in docs],
        "count": len(docs),
    }


@mcp.tool()
async def get_document_count() -> dict:
    """Get the total number of document chunks indexed."""
    if _retriever is None:
        return {"error": "Retriever not initialised"}

    total_chunks = _retriever._store.count()
    docs = _retriever._store.list_documents()
    return {
        "total_chunks": total_chunks,
        "total_documents": len(docs),
    }
