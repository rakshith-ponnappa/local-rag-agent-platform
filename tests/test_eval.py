import json
from pathlib import Path

from agent_platform.embeddings import HashingEmbedder
from agent_platform.eval_metrics import (
    hit_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from agent_platform.eval_retrieval import evaluate, load_eval_cases
from agent_platform.rag_ingest import build_chunk_records, load_documents
from agent_platform.vector_store import LocalVectorStore, VectorRecord

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hit_at_k() -> None:
    assert hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
    assert hit_at_k(["a", "b"], {"z"}, 2) == 0.0


def test_precision_and_recall_at_k() -> None:
    # 1 of the top-2 retrieved is relevant -> precision 0.5
    assert precision_at_k(["a", "b"], {"a"}, 2) == 0.5
    # retrieved 1 of 2 relevant ids -> recall 0.5
    assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert round(reciprocal_rank(["a", "b", "c"], {"c"}), 3) == 0.333
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_mean() -> None:
    assert mean([1.0, 0.0]) == 0.5
    assert mean([]) == 0.0


def test_evaluate_perfect_retrieval() -> None:
    embedder = HashingEmbedder(dim=128)
    samples = [
        ("aiops_id", "aiops is about reliable ai systems in production"),
        ("mlops_id", "mlops adds lifecycle automation and governance"),
    ]
    store = LocalVectorStore()
    store.add(
        [
            VectorRecord(chunk_id=cid, text=text, embedding=embedder.embed_one(text))
            for cid, text in samples
        ]
    )

    cases = [
        {"query": "reliable ai systems in production", "relevant_chunk_ids": ["aiops_id"]},
        {"query": "lifecycle automation and governance", "relevant_chunk_ids": ["mlops_id"]},
    ]

    report = evaluate(store, embedder, cases, k=1)

    assert report.num_cases == 2
    assert report.hit_at_k == 1.0
    assert report.mrr == 1.0


def test_eval_set_labels_exist_in_sample_corpus() -> None:
    eval_file = REPO_ROOT / "evals" / "retrieval_eval.json"
    build = json.loads(eval_file.read_text(encoding="utf-8"))["build"]
    docs = load_documents(REPO_ROOT / build["input_dir"])
    records = build_chunk_records(docs, chunk_size=build["chunk_size"], overlap=build["overlap"])
    chunk_ids = {record["chunk_id"] for record in records}

    for case in load_eval_cases(eval_file):
        missing = set(case["relevant_chunk_ids"]) - chunk_ids
        assert not missing, f"{case['query']!r} references unknown chunk ids {missing}"
