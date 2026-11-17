from pathlib import Path

from agent_platform.embeddings import HashingEmbedder
from agent_platform.vector_store import (
    LocalVectorStore,
    VectorRecord,
    cosine_similarity,
)


def test_cosine_similarity_bounds() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_query_ranks_most_similar_first() -> None:
    embedder = HashingEmbedder(dim=64)
    texts = {
        "restart the monitoring agent": "doc_agent",
        "how to bake sourdough bread": "doc_bread",
    }
    records = [
        VectorRecord(chunk_id=chunk_id, text=text, embedding=embedder.embed_one(text))
        for text, chunk_id in texts.items()
    ]
    store = LocalVectorStore()
    store.add(records)

    query = embedder.embed_one("reboot the monitoring agent")
    results = store.query(query, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "doc_agent"
    assert results[0].score >= results[1].score


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dim=32)
    record = VectorRecord(
        chunk_id="c1",
        text="hello world",
        embedding=embedder.embed_one("hello world"),
        metadata={"source": "a.md", "chunk_index": 0},
    )
    store = LocalVectorStore()
    store.add([record])

    index_file = tmp_path / "index.json"
    store.save(index_file)
    loaded = LocalVectorStore.load(index_file)

    assert len(loaded) == 1
    result = loaded.query(embedder.embed_one("hello world"), top_k=1)[0]
    assert result.chunk_id == "c1"
    assert result.metadata["source"] == "a.md"
