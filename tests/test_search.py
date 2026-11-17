from fastapi.testclient import TestClient

import agent_platform.main as main
from agent_platform.embeddings import HashingEmbedder
from agent_platform.vector_store import LocalVectorStore, VectorRecord


def _build_store(embedder: HashingEmbedder) -> LocalVectorStore:
    samples = [
        ("a1", "restart the monitoring agent on the node", "runbook.md"),
        ("a2", "sourdough bread needs a long cold proof", "recipes.md"),
    ]
    records = [
        VectorRecord(
            chunk_id=chunk_id,
            text=text,
            embedding=embedder.embed_one(text),
            metadata={"source": source, "chunk_index": 0},
        )
        for chunk_id, text, source in samples
    ]
    store = LocalVectorStore()
    store.add(records)
    return store


def test_search_returns_relevant_chunk(monkeypatch) -> None:
    embedder = HashingEmbedder(dim=128)
    monkeypatch.setattr(main, "embedder", embedder)
    monkeypatch.setattr(main, "vector_store", _build_store(embedder))

    client = TestClient(main.app)
    response = client.post("/search", json={"query": "reboot monitoring agent", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "reboot monitoring agent"
    assert len(body["hits"]) == 1
    assert body["hits"][0]["chunk_id"] == "a1"
    assert body["hits"][0]["source"] == "runbook.md"


def test_search_returns_503_when_index_empty(monkeypatch) -> None:
    monkeypatch.setattr(main, "vector_store", None)

    client = TestClient(main.app)
    response = client.post("/search", json={"query": "anything", "top_k": 3})

    assert response.status_code == 503
    assert "index is empty" in response.json()["detail"].lower()


def test_search_rejects_empty_query() -> None:
    client = TestClient(main.app)
    response = client.post("/search", json={"query": "", "top_k": 3})
    assert response.status_code == 422
