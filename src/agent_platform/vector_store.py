"""A minimal local vector database.

This is a teaching-sized version of what Chroma / Qdrant do internally:
store records that each carry an embedding, then answer nearest-neighbor
queries using cosine similarity. It brute-force scans every record, which is
perfect up to thousands of chunks. Real vector DBs add approximate indexes
(e.g. HNSW) to stay fast over millions of vectors.

Persistence is a single JSON file, so the index survives restarts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity: dot(a, b) / (|a| * |b|). Range -1..1, higher = closer."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class VectorRecord:
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, str | int] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str | int]


class LocalVectorStore:
    """In-memory vector store with JSON persistence and cosine search."""

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        dim = len(records[0].embedding)
        for record in records:
            if len(record.embedding) != dim:
                raise ValueError("all embeddings must have the same dimension")
        self._records.extend(records)

    def query(self, embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        scored = [
            SearchResult(
                chunk_id=record.chunk_id,
                text=record.text,
                score=cosine_similarity(embedding, record.embedding),
                metadata=record.metadata,
            )
            for record in self._records
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "chunk_id": record.chunk_id,
                "text": record.text,
                "embedding": record.embedding,
                "metadata": record.metadata,
            }
            for record in self._records
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalVectorStore":
        store = cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        store._records = [
            VectorRecord(
                chunk_id=item["chunk_id"],
                text=item["text"],
                embedding=[float(value) for value in item["embedding"]],
                metadata=item.get("metadata", {}),
            )
            for item in raw
        ]
        return store
