from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(input_dir: Path) -> list[tuple[Path, str]]:
    """Load .md/.txt files under ``input_dir``.

    Paths are returned relative to ``input_dir`` so chunk IDs (which hash the
    source path) are stable regardless of where the corpus lives on disk.
    """
    docs: list[tuple[Path, str]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append((path.relative_to(input_dir), text))
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def build_chunk_records(
    docs: Iterable[tuple[Path, str]],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []

    for source_path, text in docs:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            raw_id = f"{source_path}:{idx}:{chunk}".encode("utf-8")
            chunk_id = hashlib.sha1(raw_id).hexdigest()[:16]
            records.append(
                {
                    "chunk_id": chunk_id,
                    "source": str(source_path),
                    "chunk_index": idx,
                    "text": chunk,
                }
            )

    return records


def write_jsonl(records: Iterable[dict[str, str | int]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunked corpus JSONL for RAG")
    parser.add_argument("--input-dir", required=True, help="Directory with .md/.txt files")
    parser.add_argument("--output-file", required=True, help="Output JSONL path")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()

    docs = load_documents(input_dir)
    records = build_chunk_records(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    write_jsonl(records, output_file)

    print(
        f"Ingestion complete: docs={len(docs)} chunks={len(records)} output={output_file}"
    )


if __name__ == "__main__":
    main()
