"""Retrieval quality metrics.

All functions take the *ordered* list of retrieved chunk IDs (best first) and
the set of chunk IDs that are actually relevant for a query. They are pure and
deterministic, so they are easy to test and to reuse across experiments.
"""

from __future__ import annotations

from collections.abc import Iterable


def _top_k(retrieved: list[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be > 0")
    return retrieved[:k]


def hit_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant id is in the top-k, else 0.0."""
    relevant_set = set(relevant)
    return 1.0 if any(cid in relevant_set for cid in _top_k(retrieved, k)) else 0.0


def precision_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    relevant_set = set(relevant)
    top = _top_k(retrieved, k)
    if not top:
        return 0.0
    hits = sum(1 for cid in top if cid in relevant_set)
    return hits / len(top)


def recall_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of all relevant ids that appear in the top-k."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = set(_top_k(retrieved, k))
    hits = sum(1 for cid in relevant_set if cid in top)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved: list[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant result (1-based). 0.0 if none found."""
    relevant_set = set(relevant)
    for index, cid in enumerate(retrieved, start=1):
        if cid in relevant_set:
            return 1.0 / index
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
