"""FastAPI application — native RAG endpoints + OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from ragcore.config import Settings, settings as default_settings
from ragcore.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    DocumentInfo,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from ragcore.retrieval import Retriever
from ragcore.server.middleware import ObservabilityMiddleware, RateLimitMiddleware
from ragcore.store.ingest import Ingestor


# ---------------------------------------------------------------------------
# OpenAPI tag definitions
# ---------------------------------------------------------------------------

_TAGS: list[dict[str, Any]] = [
    {
        "name": "RAG",
        "description": "Core retrieval-augmented generation endpoints — search, ingest, manage documents.",
    },
    {
        "name": "Namespaces",
        "description": "Multi-tenancy support — list and query isolated knowledge-base namespaces.",
    },
    {
        "name": "OpenAI",
        "description": "OpenAI-compatible endpoints for drop-in integration with AI SDKs.",
    },
    {
        "name": "System",
        "description": "Operational endpoints: health checks.",
    },
]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    retriever: Retriever | None = None,
    ingestor: Ingestor | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create and return the FastAPI application.

    *retriever* and *ingestor* can be injected for testing; if omitted the
    module-level singletons are used.
    """
    cfg = settings or default_settings

    app = FastAPI(
        title="ragcore",
        description=(
            "RAG-as-a-service — MCP + OpenAI-compatible API.\n\n"
            "ragcore indexes documents via `/documents/upload`, then retrieves "
            "relevant chunks via `/search` (sliding-window rate-limited, LRU-cached). "
            "All responses carry `X-Request-ID` and `X-Latency-Ms` headers."
        ),
        version="1.0.0",
        openapi_tags=_TAGS,
    )

    # ------------------------------------------------------------------
    # Middleware (order matters: outermost first)
    # ------------------------------------------------------------------
    app.add_middleware(ObservabilityMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # Store on app state so route handlers can access via request.app.state
    app.state.retriever = retriever
    app.state.ingestor = ingestor
    app.state.settings = cfg

    # ------------------------------------------------------------------
    # Dependency helpers
    # ------------------------------------------------------------------

    def get_retriever() -> Retriever:
        r = app.state.retriever
        if r is None:  # pragma: no cover
            raise HTTPException(status_code=503, detail="Retriever not initialised")
        return r

    def get_ingestor() -> Ingestor:
        ing = app.state.ingestor
        if ing is None:  # pragma: no cover
            raise HTTPException(status_code=503, detail="Ingestor not initialised")
        return ing

    # ------------------------------------------------------------------
    # Native RAG endpoints
    # ------------------------------------------------------------------

    @app.get(
        "/health",
        tags=["System"],
        summary="Liveness and readiness probe",
        description=(
            "Returns `{\"status\": \"ok\"}` when the service is healthy. "
            "Also reports the document chunk count and model configuration."
        ),
        responses={
            200: {
                "description": "Service is healthy",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "version": "1.0.0",
                            "doc_count": 42,
                            "checks": {
                                "chroma": "ok",
                                "embedding_model": "all-MiniLM-L6-v2",
                                "rerank_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
                            },
                        }
                    }
                },
            }
        },
    )
    async def health(ret: Retriever = Depends(get_retriever)) -> dict[str, Any]:
        doc_count = ret._store.count()
        return {
            "status": "ok",
            "version": "1.0.0",
            "doc_count": doc_count,
            "checks": {
                "chroma": "ok",
                "embedding_provider": cfg.embedding_provider,
                "embedding_model": cfg.embedding_model,
                "rerank_model": cfg.rerank_model,
            },
        }

    @app.post(
        "/search",
        response_model=SearchResponse,
        tags=["RAG"],
        summary="Semantic search over indexed documents",
        description=(
            "Embeds the query, performs approximate nearest-neighbour search in ChromaDB, "
            "then reranks candidates with a CrossEncoder. "
            "Results are cached (LRU, 300 s TTL). "
            "Accepts an optional `namespace` query parameter to restrict search to a "
            "specific knowledge-base partition."
        ),
        responses={
            200: {
                "description": "Ranked search results",
                "content": {
                    "application/json": {
                        "example": {
                            "results": [
                                {
                                    "id": "default:abc123",
                                    "content": "Retrieval-augmented generation improves LLM accuracy.",
                                    "score": 0.92,
                                    "filename": "rag_overview.pdf",
                                    "page": 1,
                                    "chunk_index": 0,
                                    "metadata": {},
                                }
                            ],
                            "total": 1,
                            "query": "what is RAG?",
                            "latency_ms": 38.5,
                            "cache_hit": False,
                        }
                    }
                },
            },
            422: {"description": "Validation error (e.g. empty query)"},
            429: {"description": "Rate limit exceeded"},
        },
    )
    async def search(
        req: SearchRequest,
        ret: Retriever = Depends(get_retriever),
    ) -> SearchResponse:
        return await ret.search(query=req.query, top_n=req.top_n, filters=req.filters)

    @app.post(
        "/documents/upload",
        tags=["RAG"],
        summary="Ingest a document",
        description=(
            "Upload a file to be read, chunked, embedded, and stored in ChromaDB. "
            "Supported formats: `.txt`, `.pdf`, `.docx`, `.xlsx`, `.xls`, `.md`, "
            "`.json`, `.yaml`, `.yml`, `.toml`, `.csv`, `.py`, `.ts`, `.js`, `.go`, "
            "`.rs`, `.java`. "
            "Code files use a smaller chunk size (256 chars) to preserve function/class "
            "boundaries and are tagged with `file_type: code` in metadata."
        ),
        responses={
            200: {
                "description": "Document ingested successfully",
                "content": {
                    "application/json": {
                        "example": {"filename": "report.pdf", "chunks_indexed": 15, "status": "ok"}
                    }
                },
            },
            422: {"description": "Unsupported file format or empty document"},
        },
    )
    async def upload_document(
        file: UploadFile = File(...),
        ing: Ingestor = Depends(get_ingestor),
    ) -> dict[str, Any]:
        data = await file.read()
        filename = file.filename or "upload"
        try:
            chunks_indexed = ing.ingest(filename, data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"filename": filename, "chunks_indexed": chunks_indexed, "status": "ok"}

    @app.get(
        "/documents",
        response_model=list[DocumentInfo],
        tags=["RAG"],
        summary="List ingested documents",
        description=(
            "Returns all documents that have been indexed, with chunk counts and "
            "ingestion timestamps. Accepts an optional `namespace` query parameter "
            "to list only documents in a specific partition."
        ),
        responses={
            200: {
                "description": "List of document metadata",
                "content": {
                    "application/json": {
                        "example": [
                            {"filename": "report.pdf", "chunks": 15, "added_at": "2024-01-01T00:00:00+00:00"}
                        ]
                    }
                },
            }
        },
    )
    async def list_documents(
        ret: Retriever = Depends(get_retriever),
    ) -> list[DocumentInfo]:
        return ret._store.list_documents()

    @app.delete(
        "/documents/{filename}",
        tags=["RAG"],
        summary="Delete a document",
        description="Remove all chunks for the given filename from the vector store.",
        responses={
            200: {
                "description": "Deletion result",
                "content": {
                    "application/json": {
                        "example": {"filename": "report.pdf", "deleted": 15}
                    }
                },
            }
        },
    )
    async def delete_document(
        filename: str,
        ret: Retriever = Depends(get_retriever),
    ) -> dict[str, Any]:
        deleted = ret._store.delete_by_filename(filename)
        return {"filename": filename, "deleted": deleted}

    # ------------------------------------------------------------------
    # Namespace endpoints
    # ------------------------------------------------------------------

    @app.get(
        "/namespaces",
        tags=["Namespaces"],
        summary="List all distinct namespaces",
        description=(
            "Queries ChromaDB metadata to return every distinct namespace value "
            "present in the collection. Namespaces allow multiple isolated knowledge "
            "bases to coexist in a single ChromaDB instance."
        ),
        responses={
            200: {
                "description": "Sorted list of namespace strings",
                "content": {
                    "application/json": {
                        "example": {"namespaces": ["default", "project-alpha", "support-docs"]}
                    }
                },
            }
        },
    )
    async def list_namespaces(
        ret: Retriever = Depends(get_retriever),
    ) -> dict[str, Any]:
        namespaces = ret._store.list_namespaces()
        return {"namespaces": namespaces}

    # ------------------------------------------------------------------
    # OpenAI-compatible endpoints
    # ------------------------------------------------------------------

    @app.get(
        "/v1/models",
        tags=["OpenAI"],
        summary="List available models (OpenAI-compatible)",
        description="Returns the ragcore pseudo-model in OpenAI model-list format.",
        responses={
            200: {
                "description": "Model list",
                "content": {
                    "application/json": {
                        "example": {
                            "object": "list",
                            "data": [{"id": "ragcore", "object": "model", "created": 1700000000, "owned_by": "ragcore"}],
                        }
                    }
                },
            }
        },
    )
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "ragcore",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ragcore",
                }
            ],
        }

    @app.post(
        "/v1/embeddings",
        tags=["OpenAI"],
        summary="Create embeddings (OpenAI-compatible)",
        description=(
            "Accepts `{\"input\": \"text\" | [\"text1\", \"text2\"], \"model\": \"...\"}` "
            "and returns dense vectors in OpenAI embedding format using ragcore's "
            "configured sentence-transformer model."
        ),
        responses={
            200: {
                "description": "Embedding vectors",
                "content": {
                    "application/json": {
                        "example": {
                            "object": "list",
                            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, -0.2, 0.3]}],
                            "model": "ragcore",
                            "usage": {"prompt_tokens": 2, "total_tokens": 2},
                        }
                    }
                },
            }
        },
    )
    async def create_embeddings(
        body: dict,
        ret: Retriever = Depends(get_retriever),
    ) -> dict[str, Any]:
        import asyncio
        from functools import partial

        raw_input = body.get("input", "")
        texts: list[str] = [raw_input] if isinstance(raw_input, str) else list(raw_input)

        loop = asyncio.get_event_loop()
        embeddings_array = await loop.run_in_executor(
            None,
            partial(
                ret._embed_model.encode,
                texts,
                show_progress_bar=False,
            ),
        )

        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": emb.tolist() if hasattr(emb, "tolist") else list(emb),
            }
            for i, emb in enumerate(embeddings_array)
        ]

        return {
            "object": "list",
            "data": data,
            "model": body.get("model", "ragcore"),
            "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
        }

    @app.post(
        "/v1/chat/completions",
        tags=["OpenAI"],
        summary="Chat completions — RAG context retrieval (OpenAI-compatible)",
        description=(
            "OpenAI-compatible chat completions endpoint.\n\n"
            "**Important:** ragcore does NOT call any LLM. "
            "The last user message is used as the RAG search query. "
            "Retrieved chunks are JSON-serialised and returned in "
            "`choices[0].message.content` so the calling AI can consume them. "
            "The raw typed results are also available in the bonus `rag_results` field."
        ),
        responses={
            200: {
                "description": "Retrieved context in OpenAI chat-completion shape",
                "content": {
                    "application/json": {
                        "example": {
                            "id": "ragcore-a1b2c3d4",
                            "object": "chat.completion",
                            "model": "ragcore",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "{\"query\": \"...\", \"results\": [], \"total\": 0, \"latency_ms\": 12.3, \"_note\": \"This is retrieved context, not an LLM-generated answer.\"}",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {},
                            "rag_results": [],
                        }
                    }
                },
            },
            422: {"description": "No user message provided"},
        },
    )
    async def chat_completions(
        req: ChatCompletionRequest,
        ret: Retriever = Depends(get_retriever),
    ) -> ChatCompletionResponse:
        # Extract the last user message as the search query
        user_messages = [m for m in req.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=422,
                detail="At least one user message is required",
            )
        query = user_messages[-1].content

        response = await ret.search(query=query, top_n=req.top_n)

        # Serialise results to JSON string — the calling AI decides what to do with them
        content_json = json.dumps(
            {
                "query": query,
                "results": [r.model_dump() for r in response.results],
                "total": response.total,
                "latency_ms": response.latency_ms,
                "_note": "This is retrieved context, not an LLM-generated answer.",
            },
            ensure_ascii=False,
        )

        return ChatCompletionResponse(
            id=f"ragcore-{uuid.uuid4().hex[:8]}",
            model=req.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content_json},
                    "finish_reason": "stop",
                }
            ],
            usage={},
            rag_results=response.results,
        )

    return app
