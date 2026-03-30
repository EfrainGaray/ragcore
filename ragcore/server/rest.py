"""FastAPI application — native RAG endpoints + OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
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
from ragcore.store.ingest import Ingestor


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
        description="RAG-as-a-service — MCP + OpenAI-compatible API",
        version="1.0.0",
    )

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

    @app.get("/health")
    async def health(ret: Retriever = Depends(get_retriever)) -> dict[str, Any]:
        """Liveness + readiness probe."""
        doc_count = ret._store.count()
        return {
            "status": "ok",
            "version": "1.0.0",
            "doc_count": doc_count,
            "checks": {
                "chroma": "ok",
                "embedding_model": cfg.embedding_model,
                "rerank_model": cfg.rerank_model,
            },
        }

    @app.post("/search", response_model=SearchResponse)
    async def search(
        req: SearchRequest,
        ret: Retriever = Depends(get_retriever),
    ) -> SearchResponse:
        """Main RAG search endpoint — embed query, vector search, rerank."""
        return await ret.search(query=req.query, top_n=req.top_n, filters=req.filters)

    @app.post("/documents/upload")
    async def upload_document(
        file: UploadFile = File(...),
        ing: Ingestor = Depends(get_ingestor),
    ) -> dict[str, Any]:
        """Ingest a document: read, chunk, embed, store."""
        data = await file.read()
        filename = file.filename or "upload"
        try:
            chunks_indexed = ing.ingest(filename, data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"filename": filename, "chunks_indexed": chunks_indexed, "status": "ok"}

    @app.get("/documents", response_model=list[DocumentInfo])
    async def list_documents(
        ret: Retriever = Depends(get_retriever),
    ) -> list[DocumentInfo]:
        """List all ingested documents with chunk counts."""
        return ret._store.list_documents()

    @app.delete("/documents/{filename}")
    async def delete_document(
        filename: str,
        ret: Retriever = Depends(get_retriever),
    ) -> dict[str, Any]:
        """Delete all chunks for *filename*."""
        deleted = ret._store.delete_by_filename(filename)
        return {"filename": filename, "deleted": deleted}

    # ------------------------------------------------------------------
    # OpenAI-compatible endpoints
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """OpenAI-compatible model list endpoint."""
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

    @app.post("/v1/embeddings")
    async def create_embeddings(
        body: dict,
        ret: Retriever = Depends(get_retriever),
    ) -> dict[str, Any]:
        """OpenAI-compatible embeddings endpoint.

        Accepts ``{"input": "text" | ["text1", "text2"], "model": "..."}``
        and returns vectors in OpenAI format.
        """
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
                "embedding": emb.tolist(),
            }
            for i, emb in enumerate(embeddings_array)
        ]

        return {
            "object": "list",
            "data": data,
            "model": body.get("model", "ragcore"),
            "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        ret: Retriever = Depends(get_retriever),
    ) -> ChatCompletionResponse:
        """OpenAI-compatible chat completions endpoint.

        IMPORTANT: ragcore does NOT call any LLM.
        The last user message is used as the RAG query.
        The retrieved chunks are serialised to JSON and returned in
        choices[0].message.content so that the *calling* AI can consume them.
        """
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
