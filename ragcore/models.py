"""Pydantic schemas for ragcore."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """One retrieved chunk."""

    id: str
    content: str
    score: float          # rerank score
    filename: str
    page: int | str
    chunk_index: int
    metadata: dict = {}


class SearchRequest(BaseModel):
    """What any AI sends to /search."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_n: int | None = None   # override default
    filters: dict = {}         # optional metadata filters


class SearchResponse(BaseModel):
    """Response from /search."""

    results: list[SearchResult]
    total: int
    query: str
    latency_ms: float


class DocumentInfo(BaseModel):
    """Metadata about an ingested document."""

    filename: str
    chunks: int
    added_at: str


# ---------------------------------------------------------------------------
# OpenAI-compatible schemas
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    ragcore does NOT call an LLM. The last user message is used as the RAG
    query; retrieved chunks are returned in the response content as JSON.
    """

    model: str = "ragcore"
    messages: list[Message]
    top_n: int | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-shaped response whose content is JSON-serialised retrieved chunks.

    This is intentionally NOT an LLM answer — it is raw retrieved context so
    that the calling AI can decide what to do with it.
    """

    id: str
    object: str = "chat.completion"
    model: str = "ragcore"
    choices: list[dict]
    usage: dict = {}
    rag_results: list[SearchResult] = []  # bonus field: typed results alongside
