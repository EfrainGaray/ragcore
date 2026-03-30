"""Query Expansion (RAG-Fusion) for ragcore."""

from __future__ import annotations

from loguru import logger

from ragcore.embedding import _KNOWN_ALIASES


class QueryExpander:
    """Generates alternative query phrasings using an OpenAI-compatible LLM."""

    def __init__(
        self,
        llm_url: str = "",
        llm_key: str = "",
        llm_model: str = "gpt-4o-mini",
        n: int = 3,
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
        self._n = n
        self._headers = {
            "Authorization": f"Bearer {llm_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=30.0)

    def expand(self, query: str, n: int | None = None) -> list[str]:
        """Return [query] + N alternative phrasings.

        First element is always the original query.
        Falls back to [query] alone on any error.
        """
        effective_n = n if n is not None else self._n
        prompt = (
            f"Generate {effective_n} alternative phrasings of this search query. "
            f"Return one per line, no numbering: {query}"
        )
        try:
            response = self._client.post(
                self._endpoint,
                headers=self._headers,
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            data = response.json()
            raw: str = data["choices"][0]["message"]["content"]
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            # Deduplicate: skip lines that match the original query
            alternatives = [l for l in lines if l != query][: effective_n]
            result = [query] + alternatives
            logger.debug(
                "QueryExpander produced {} variants for query={!r}", len(result), query
            )
            return result
        except Exception as exc:
            logger.warning("QueryExpander failed ({}); using original query only", exc)
            return [query]


def build_expander(settings) -> QueryExpander | None:
    """Factory — return a QueryExpander instance if query_expansion_enabled, else None."""
    if not settings.query_expansion_enabled:
        return None
    # Shares LLM config with HyDE
    if not settings.hyde_llm_url:
        logger.warning(
            "Query expansion enabled but HYDE_LLM_URL not set; expansion disabled"
        )
        return None
    return QueryExpander(
        llm_url=settings.hyde_llm_url,
        llm_key=settings.hyde_llm_key,
        llm_model=settings.hyde_llm_model,
        n=settings.query_expansion_count,
    )
