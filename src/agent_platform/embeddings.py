"""Embedding services.

An *embedding* turns text into a fixed-length vector of numbers that captures
its meaning. Similar meaning -> vectors that point in a similar direction.

We define a small ``Embedder`` interface so the rest of the app never cares
which implementation is used:

- ``HashingEmbedder``: deterministic, offline, no model needed. Great for tests
  and local dev. It hashes each word into a fixed-size vector (a "bag of hashed
  tokens"), so texts that share words get similar vectors.
- ``OllamaEmbedder``: calls a real local embedding model via Ollama. Use this in
  production for true semantic understanding.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

import httpx

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _l2_normalize(vector: list[float]) -> list[float]:
    """Scale a vector to length 1 so cosine similarity == dot product."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class Embedder(Protocol):
    """Anything that can turn texts into vectors of equal length."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        ...


class HashingEmbedder:
    """Deterministic, dependency-free embedder for offline dev and tests.

    Each word is hashed into one of ``dim`` buckets and its count is added
    there. The vector is then L2-normalized. No network, no model, fully
    reproducible.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _tokenize(text):
            vector[self._bucket(token)] += 1.0
        return _l2_normalize(vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]


class OllamaEmbedder:
    """Real embeddings from a local Ollama embedding model.

    Pull a model first, for example: ``ollama pull nomic-embed-text``.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str | None = None,
        dim: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.timeout = timeout
        # dim is discovered from the first response if not provided.
        self.dim = dim or 0

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for text in texts:
                    response = client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                    )
                    response.raise_for_status()
                    embedding = response.json().get("embedding")
                    if not embedding:
                        raise EmbeddingError("Ollama returned an empty embedding")
                    vectors.append(_l2_normalize([float(value) for value in embedding]))
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embedding call failed: {exc}") from exc

        if vectors:
            self.dim = len(vectors[0])
        return vectors


class EmbeddingError(Exception):
    pass
