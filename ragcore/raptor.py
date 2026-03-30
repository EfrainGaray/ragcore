"""RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval."""
from __future__ import annotations

import uuid
from loguru import logger
from ragcore.embedding import _KNOWN_ALIASES


class RaptorIndexer:
    """Builds a tree of summary chunks over leaf chunks.

    Level 0 = original leaf chunks (already stored, not returned)
    Level 1 = summaries of groups of leaf chunks
    Level 2 = summaries of level-1 summaries
    ...

    Returns summary chunks to be embedded+stored by the caller.
    """

    def __init__(
        self,
        embed_model,
        llm_url: str = "",
        llm_key: str = "",
        llm_model: str = "gpt-4o-mini",
        levels: int = 3,
        group_size: int = 5,      # how many chunks to group per summary
        *,
        api_url: str = "",
        api_key: str = "",
        model: str = "",
    ) -> None:
        # Accept both naming conventions
        url = api_url or llm_url
        key = api_key or llm_key
        mdl = model or llm_model or "gpt-4o-mini"

        self._embed_model = embed_model
        self._levels = levels
        self._group_size = group_size
        self._llm_model = mdl
        self._llm_key = key
        self._endpoint = ""

        if url:
            import httpx
            base = _KNOWN_ALIASES.get(url.lower(), url)
            self._endpoint = base.rstrip("/") + "/chat/completions"
            self._headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            self._http = httpx.Client(timeout=30.0)
        else:
            self._http = None

    def _summarize(self, texts: list[str]) -> str:
        """Summarize a group of texts. Falls back to truncated concatenation on error."""
        combined = "\n\n".join(texts)
        if not self._endpoint or self._http is None:
            # No LLM — use first 512 chars as "summary"
            return combined[:512]

        prompt = f"Write a concise summary of the following passages:\n\n{combined[:3000]}"
        try:
            response = self._http.post(
                self._endpoint,
                headers=self._headers,
                json={
                    "model": self._llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.3,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            return content if content else combined[:512]
        except Exception as exc:
            logger.warning("RAPTOR summarize failed ({}); using concatenation", exc)
            return combined[:512]

    def build_tree(self, chunks: list[dict]) -> list[dict]:
        """Build RAPTOR summary tree over chunks.

        chunks: list of chunk dicts with at least "content", "filename", "page"
        Returns: list of summary chunk dicts (WITHOUT embeddings — caller embeds them)
                 Each has: id, content, filename, page, chunk_index, file_type,
                           chunk_type="raptor_summary", raptor_level=N, child_ids=[...]
        """
        if not chunks:
            return []

        all_summary_chunks = []
        current_level = chunks

        for level in range(1, self._levels + 1):
            if len(current_level) <= 1:
                break  # No point summarizing a single chunk

            level_summaries = []

            # Group chunks
            groups = [
                current_level[i:i + self._group_size]
                for i in range(0, len(current_level), self._group_size)
            ]

            for group in groups:
                texts = [c["content"] for c in group]
                summary_text = self._summarize(texts)

                if not summary_text.strip():
                    continue

                # Use filename/page from first chunk in group
                first = group[0]
                child_ids = [c["id"] for c in group]

                summary_chunk = {
                    "id": str(uuid.uuid4()),
                    "content": summary_text,
                    "filename": first.get("filename", ""),
                    "page": first.get("page", 0),
                    "chunk_index": 0,
                    "file_type": first.get("file_type", "document"),
                    "chunk_type": "raptor_summary",
                    "raptor_level": level,
                    "child_ids": child_ids,
                }
                level_summaries.append(summary_chunk)
                all_summary_chunks.append(summary_chunk)

            logger.debug(
                "RAPTOR level {}: {} groups → {} summaries",
                level, len(groups), len(level_summaries)
            )

            current_level = level_summaries

        return all_summary_chunks


def build_raptor(settings, embed_model) -> RaptorIndexer | None:
    """Factory — returns RaptorIndexer if raptor_enabled, else None."""
    if not getattr(settings, "raptor_enabled", False):
        return None
    return RaptorIndexer(
        embed_model=embed_model,
        llm_url=getattr(settings, "raptor_llm_url", "") or getattr(settings, "hyde_llm_url", ""),
        llm_key=getattr(settings, "raptor_llm_key", "") or getattr(settings, "hyde_llm_key", ""),
        llm_model=getattr(settings, "raptor_llm_model", "gpt-4o-mini"),
        levels=getattr(settings, "raptor_levels", 3),
    )
