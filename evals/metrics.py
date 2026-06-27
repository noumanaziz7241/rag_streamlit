"""Evaluation metrics for RAG quality."""

from __future__ import annotations

from typing import Iterable, List, Sequence


def recall_at_k(retrieved_sources: Sequence[str], expected_source: str, k: int = 5) -> float:
    """Return 1.0 if expected source appears in top-k retrieved sources, else 0.0."""
    top = [source.lower() for source in retrieved_sources[:k]]
    return 1.0 if any(expected_source.lower() in source for source in top) else 0.0


def keyword_faithfulness(answer: str, expected_keywords: Iterable[str]) -> float:
    """Fraction of expected keywords present in the generated answer."""
    keywords = [keyword.lower() for keyword in expected_keywords if keyword.strip()]
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for keyword in keywords if keyword in answer_lower)
    return hits / len(keywords)


def mean_score(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
