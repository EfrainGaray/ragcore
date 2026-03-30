"""Tests for ragcore.evaluation — RAGEvaluator and EvalResult."""

from __future__ import annotations

import pytest
from tests.conftest import _FakeSentenceTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evaluator(llm_url: str = ""):
    from ragcore.evaluation import RAGEvaluator

    return RAGEvaluator(
        embed_model=_FakeSentenceTransformer(),
        llm_url=llm_url,
        llm_key="",
        llm_model="gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_evaluator_returns_eval_result():
    """evaluate() must return an object with faithfulness, answer_relevance,
    context_recall, and context_precision attributes."""
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        query="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
        contexts=["RAG is a technique that combines retrieval with generation."],
    )
    assert hasattr(result, "faithfulness")
    assert hasattr(result, "answer_relevance")
    assert hasattr(result, "context_recall")
    assert hasattr(result, "context_precision")


def test_answer_relevance_high_for_matching_answer():
    """When query and answer are identical the relevance score must be >= 0.9."""
    evaluator = _make_evaluator()
    text = "What is retrieval augmented generation?"
    result = evaluator.evaluate(
        query=text,
        answer=text,
        contexts=["Some context."],
    )
    assert result.answer_relevance >= 0.9


def test_context_precision_scores_relevant_contexts():
    """A context that closely matches the query should yield context_precision > 0."""
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        query="What is RAG?",
        answer="RAG is retrieval augmented generation.",
        contexts=["RAG is a method that retrieves documents before generating an answer."],
    )
    assert result.context_precision > 0.0


def test_evaluation_without_ground_truth():
    """When ground_truth is not provided, context_recall must be None or 0.0."""
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        query="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
        contexts=["RAG retrieves documents to augment generation."],
        ground_truth=None,
    )
    assert result.context_recall is None or result.context_recall == 0.0


def test_evaluator_all_scores_in_range():
    """Every non-None score must be in [0.0, 1.0]."""
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        query="Explain neural networks.",
        answer="Neural networks are models inspired by the brain.",
        contexts=[
            "Neural networks consist of layers of interconnected nodes.",
            "Deep learning uses many-layered neural networks.",
        ],
        ground_truth="Neural networks are computational models inspired by the brain.",
    )
    for field in ("faithfulness", "answer_relevance", "context_precision"):
        value = getattr(result, field)
        if value is not None:
            assert 0.0 <= value <= 1.0, f"{field}={value} out of range"
    if result.context_recall is not None:
        assert 0.0 <= result.context_recall <= 1.0
