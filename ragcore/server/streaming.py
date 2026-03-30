"""SSE streaming for /v1/chat/completions with stream=true."""
from __future__ import annotations

import json
from typing import AsyncGenerator
from loguru import logger


async def stream_chat_completion(
    query: str,
    context: str,
    llm_url: str,
    llm_key: str,
    llm_model: str = "gpt-4o-mini",
) -> AsyncGenerator[str, None]:
    """Yield SSE lines from an OpenAI-compatible streaming chat completion.

    Each yielded string is a complete SSE line: `data: {...}\\n\\n`
    The final line is `data: [DONE]\\n\\n`

    Falls back to yielding context as a single non-streaming chunk if:
    - llm_url is empty
    - The API call fails
    """
    if not llm_url:
        # No LLM configured — yield context as single chunk
        chunk = {
            "choices": [{"delta": {"content": context}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    from ragcore.embedding import _KNOWN_ALIASES
    base = _KNOWN_ALIASES.get(llm_url.lower(), llm_url)
    endpoint = base.rstrip("/") + "/chat/completions"

    system_prompt = "You are a helpful assistant. Answer the question using only the provided context."
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {llm_key}",
        "Content-Type": "application/json",
    }

    try:
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if raw == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    # Pass through as-is
                    yield f"data: {raw}\n\n"
    except Exception as exc:
        logger.warning("Streaming LLM failed ({}); falling back to context", exc)
        # Fallback: yield context as single chunk
        chunk = {
            "choices": [{"delta": {"content": context}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
