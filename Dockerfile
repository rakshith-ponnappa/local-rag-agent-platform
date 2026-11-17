FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Build the sample index at image build time (offline, hashing embedder).
COPY sample_docs ./sample_docs
RUN python -m agent_platform.rag_ingest --input-dir sample_docs --output-file artifacts/chunks.jsonl --chunk-size 400 --overlap 80 \
 && python -m agent_platform.index_corpus --chunks-file artifacts/chunks.jsonl --index-file artifacts/index.json --embedder hashing

RUN useradd --create-home --uid 10001 app
USER app

ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    INDEX_FILE=/app/artifacts/index.json

EXPOSE 8000
CMD ["uvicorn", "agent_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]
