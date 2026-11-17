from pathlib import Path

from agent_platform.rag_ingest import (
    build_chunk_records,
    chunk_text,
    load_documents,
    write_jsonl,
)


def test_chunk_text_splits_with_overlap() -> None:
    text = "A" * 50
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) == 4
    assert chunks[0] == "A" * 20


def test_build_chunk_records_creates_metadata() -> None:
    docs = [(Path("notes.md"), "hello world " * 20)]
    records = build_chunk_records(docs, chunk_size=40, overlap=10)
    assert len(records) > 0
    assert "chunk_id" in records[0]
    assert records[0]["source"] == "notes.md"


def test_load_documents_and_write_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "docs"
    src.mkdir()
    (src / "a.md").write_text("alpha", encoding="utf-8")
    (src / "b.txt").write_text("beta", encoding="utf-8")
    (src / "c.json").write_text("ignored", encoding="utf-8")

    docs = load_documents(src)
    assert len(docs) == 2
    # Paths are relative to the input dir so chunk IDs don't depend on checkout location.
    assert [str(path) for path, _ in docs] == ["a.md", "b.txt"]

    out = tmp_path / "chunks.jsonl"
    records = build_chunk_records(docs, chunk_size=10, overlap=2)
    write_jsonl(records, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
