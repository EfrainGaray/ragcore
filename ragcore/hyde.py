"""HyDE — Hypothetical Document Embeddings for ragcore."""

from __future__ import annotations

from loguru import logger

from ragcore.embedding import _KNOWN_ALIASES


class HyDE:
    """Calls an OpenAI-compatible /v1/chat/completions to generate a hypothetical document.

    The generated passage is then embedded instead of the raw query, bringing
    the query vector closer to actual document vectors in embedding space.
    """

    def __init__(
        self,
        llm_url: str = "",
        llm_key: str = "",
        llm_model: str = "gpt-4o-mini",
        *,
        api_url: str = "",
        api_key: str = "",
        model: str = "",
    ) -> None:
        import httpx

        # Accept both naming conventions
        url = api_url or llm_url
        key = api_key or llm_key
        mdl = model or llm_model or "gpt-4o-mini"

        base = _KNOWN_ALIASES.get(url.lower(), url)
        self._endpoint = base.rstrip("/") + "/chat/completions"
        self._model = mdl
        self._headers = {
            "Authorization": f"Bearer {llm_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=30.0)

    def generate(self, query: str) -> str:
        """Return a short hypothetical passage that would answer the query.

        Designed to be called via loop.run_in_executor (synchronous / blocking).
        Falls back to returning the original query on any error.
        """
        prompt = (
            f"Write a short passage that directly answers: {query}. "
            "Be factual and concise."
        )
        try:
            response = self._client.post(
                self._endpoint,
                headers=self._headers,
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            data = response.json()
            content: str = data["choices"][0]["message"]["content"]
            if not content:
                return query
            logger.debug("HyDE generated {} chars for query={!r}", len(content), query)
            return content
        except Exception as exc:
            logger.warning("HyDE generation failed ({}); falling back to raw query", exc)
            return query


def build_hyde(settings) -> HyDE | None:
    """Factory — return a HyDE instance if hyde_enabled, else None."""
    if not settings.hyde_enabled:
        return None
    if not settings.hyde_llm_url:
        logger.warning("HyDE enabled but HYDE_LLM_URL not set; HyDE disabled")
        return None
    return HyDE(
        llm_url=settings.hyde_llm_url,
        llm_key=settings.hyde_llm_key,
        llm_model=settings.hyde_llm_model,
    )
