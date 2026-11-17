"""Index a chunked corpus into the local vector store.

Reads the JSONL produced by ``rag_ingest.py``, embeds every chunk, and writes
a persistent vector index. Run after ingestion:

    python -m agent_platform.index_corpus \
        --chunks-file ./artifacts/chunks.jsonl \
        --index-file ./artifacts/index.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_platform.embeddings import Embedder, HashingEmbedder, OllamaEmbedder
from agent_platform.vector_store import LocalVectorStore, VectorRecord


def load_chunks(chunks_file: Path) -> list[dict]:
    records: list[dict] = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_store(chunks: list[dict], embedder: Embedder) -> LocalVectorStore:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed(texts)

    records = [
        VectorRecord(
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            embedding=embedding,
            metadata={
                "source": chunk.get("source", ""),
                "chunk_index": chunk.get("chunk_index", 0),
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    store = LocalVectorStore()
    store.add(records)
    return store


def make_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name == "ollama":
        return OllamaEmbedder()
    raise ValueError(f"unknown embedder: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a chunked corpus into a vector store")
    parser.add_argument("--chunks-file", required=True, help="JSONL from rag_ingest.py")
    parser.add_argument("--index-file", required=True, help="Output index JSON path")
    parser.add_argument(
        "--embedder",
        choices=["hashing", "ollama"],
        default="hashing",
        help="hashing = offline/deterministic; ollama = real local model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks_file = Path(args.chunks_file).expanduser().resolve()
    index_file = Path(args.index_file).expanduser().resolve()

    chunks = load_chunks(chunks_file)
    embedder = make_embedder(args.embedder)
    store = build_store(chunks, embedder)
    store.save(index_file)

    print(
        f"Indexing complete: chunks={len(chunks)} vectors={len(store)} "
        f"embedder={args.embedder} output={index_file}"
    )


if __name__ == "__main__":
    main()
