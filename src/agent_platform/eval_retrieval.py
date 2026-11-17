"""Evaluate retrieval quality against a labeled eval set.

Loads a vector index and a JSON eval file, runs the retriever for every query,
and reports averaged Hit@k, Precision@k, Recall@k, and MRR. Use it to compare
embedders, chunk sizes, or top_k settings over time.

    python -m agent_platform.eval_retrieval \
        --index-file ./artifacts/index.json \
        --eval-file ./evals/retrieval_eval.json \
        --k 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from agent_platform.embeddings import Embedder, HashingEmbedder, OllamaEmbedder
from agent_platform.eval_metrics import (
    hit_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from agent_platform.vector_store import LocalVectorStore


@dataclass
class EvalReport:
    num_cases: int
    k: int
    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr: float


def load_eval_cases(eval_file: Path) -> list[dict]:
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if not cases:
        raise ValueError("eval file has no cases")
    return cases


def evaluate(
    store: LocalVectorStore,
    embedder: Embedder,
    cases: list[dict],
    k: int,
) -> EvalReport:
    hits: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    rr: list[float] = []

    for case in cases:
        query = case["query"]
        relevant = case["relevant_chunk_ids"]
        query_embedding = embedder.embed_one(query)
        results = store.query(query_embedding, top_k=k)
        retrieved_ids = [result.chunk_id for result in results]

        hits.append(hit_at_k(retrieved_ids, relevant, k))
        precisions.append(precision_at_k(retrieved_ids, relevant, k))
        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        rr.append(reciprocal_rank(retrieved_ids, relevant))

    return EvalReport(
        num_cases=len(cases),
        k=k,
        hit_at_k=mean(hits),
        precision_at_k=mean(precisions),
        recall_at_k=mean(recalls),
        mrr=mean(rr),
    )


def make_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name == "ollama":
        return OllamaEmbedder()
    raise ValueError(f"unknown embedder: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--index-file", required=True, help="Vector index JSON")
    parser.add_argument("--eval-file", required=True, help="Labeled eval JSON")
    parser.add_argument("--k", type=int, default=3, help="Top-k cutoff for metrics")
    parser.add_argument("--embedder", choices=["hashing", "ollama"], default="hashing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = LocalVectorStore.load(Path(args.index_file).expanduser().resolve())
    cases = load_eval_cases(Path(args.eval_file).expanduser().resolve())
    embedder = make_embedder(args.embedder)

    report = evaluate(store, embedder, cases, k=args.k)

    print("Retrieval evaluation")
    print(f"  cases        : {report.num_cases}")
    print(f"  k            : {report.k}")
    print(f"  Hit@{report.k}       : {report.hit_at_k:.3f}")
    print(f"  Precision@{report.k} : {report.precision_at_k:.3f}")
    print(f"  Recall@{report.k}    : {report.recall_at_k:.3f}")
    print(f"  MRR          : {report.mrr:.3f}")


if __name__ == "__main__":
    main()
