"""RAG evaluation metrics (RAGAS-inspired)."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from loguru import logger
from ragcore.embedding import _KNOWN_ALIASES


class EvalResult(BaseModel):
    """Evaluation scores for a single RAG query."""
    faithfulness: float           # 0.0-1.0 — answer grounded in context
    answer_relevance: float       # 0.0-1.0 — answer relevant to query
    context_precision: float      # 0.0-1.0 — contexts relevant to query
    context_recall: float | None  # 0.0-1.0 — ground_truth covered, None if no GT


class RAGEvaluator:
    """Compute RAGAS-inspired metrics for RAG evaluation.

    - answer_relevance: cosine similarity between answer embedding and query embedding
    - context_precision: mean cosine similarity of each context to the query
    - faithfulness: if LLM configured, use LLM; else heuristic (overlap between answer and contexts)
    - context_recall: if ground_truth provided, cosine sim between GT and best context
    """

    def __init__(
        self,
        embed_model,
        llm_url: str = "",
        llm_key: str = "",
        llm_model: str = "gpt-4o-mini",
        *,
        api_url: str = "",
        api_key: str = "",
        model: str = "",
    ) -> None:
        url = api_url or llm_url
        key = api_key or llm_key
        mdl = model or llm_model or "gpt-4o-mini"

        self._embed_model = embed_model
        self._llm_model = mdl

        self._endpoint = ""
        self._headers = {}
        self._http = None

        if url:
            import httpx
            base = _KNOWN_ALIASES.get(url.lower(), url)
            self._endpoint = base.rstrip("/") + "/chat/completions"
            self._headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            self._http = httpx.Client(timeout=30.0)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _embed_one(self, text: str) -> np.ndarray:
        raw = self._embed_model.encode([text], show_progress_bar=False)
        row = raw[0]
        if hasattr(row, "tolist"):
            return np.array(row)
        return np.array(list(row))

    def _embed_many(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        raw = self._embed_model.encode(texts, show_progress_bar=False)
        result = []
        for row in raw:
            if hasattr(row, "tolist"):
                result.append(np.array(row))
            else:
                result.append(np.array(list(row)))
        return result

    def _compute_answer_relevance(self, query: str, answer: str) -> float:
        q_emb = self._embed_one(query)
        a_emb = self._embed_one(answer)
        return max(0.0, min(1.0, self._cosine(q_emb, a_emb)))

    def _compute_context_precision(self, query: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        q_emb = self._embed_one(query)
        c_embs = self._embed_many(contexts)
        sims = [max(0.0, self._cosine(q_emb, c)) for c in c_embs]
        return float(np.mean(sims))

    def _compute_faithfulness_heuristic(self, answer: str, contexts: list[str]) -> float:
        """Heuristic: proportion of answer words appearing in contexts."""
        if not contexts or not answer.strip():
            return 0.0
        context_text = " ".join(contexts).lower()
        answer_words = [w.lower() for w in answer.split() if len(w) > 3]
        if not answer_words:
            return 1.0
        found = sum(1 for w in answer_words if w in context_text)
        return float(found) / len(answer_words)

    def _compute_faithfulness_llm(self, answer: str, contexts: list[str]) -> float:
        """Use LLM to score faithfulness. Falls back to heuristic on error."""
        context_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = (
            f"Given these contexts:\n{context_text}\n\n"
            f"Rate how faithful this answer is to the contexts on a scale of 0.0 to 1.0. "
            f"Answer only with a number:\n{answer}"
        )
        try:
            response = self._http.post(
                self._endpoint,
                headers=self._headers,
                json={
                    "model": self._llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            return max(0.0, min(1.0, float(raw)))
        except Exception as exc:
            logger.warning("Faithfulness LLM eval failed ({}); using heuristic", exc)
            return self._compute_faithfulness_heuristic(answer, contexts)

    def _compute_context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """Cosine similarity between ground_truth embedding and best context embedding."""
        if not contexts:
            return 0.0
        gt_emb = self._embed_one(ground_truth)
        c_embs = self._embed_many(contexts)
        sims = [self._cosine(gt_emb, c) for c in c_embs]
        return max(0.0, min(1.0, max(sims)))

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvalResult:
        """Compute all metrics. Returns EvalResult."""
        answer_relevance = self._compute_answer_relevance(query, answer)
        context_precision = self._compute_context_precision(query, contexts)

        if self._endpoint and self._http:
            faithfulness = self._compute_faithfulness_llm(answer, contexts)
        else:
            faithfulness = self._compute_faithfulness_heuristic(answer, contexts)

        context_recall = None
        if ground_truth is not None:
            context_recall = self._compute_context_recall(ground_truth, contexts)

        result = EvalResult(
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(answer_relevance, 4),
            context_precision=round(context_precision, 4),
            context_recall=round(context_recall, 4) if context_recall is not None else None,
        )
        logger.debug("RAG eval: {}", result)
        return result
