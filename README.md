# local-rag-agent-platform

[![CI](https://github.com/rakshith-ponnappa/local-rag-agent-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/rakshith-ponnappa/local-rag-agent-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A **local-first retrieval (RAG) service with retrieval evals, and no cloud keys needed.** It ingests
Markdown into chunks, embeds them, stores them in a small dependency-free vector store, and serves
semantic search over FastAPI. A local LLM endpoint is served through [Ollama](https://ollama.com).
Retrieval quality is measured with **Hit@k, Precision@k, Recall@k and MRR** on a labelled eval set,
in CI on every push.

It's built to make every moving part of a RAG pipeline visible and testable before swapping in
production components (a managed vector DB, hosted embeddings, a gateway).

## Architecture

```mermaid
flowchart LR
  subgraph Offline["Offline pipeline"]
    D[sample_docs/*.md] --> I[rag_ingest<br/>chunk 400 / overlap 80]
    I --> C[(chunks.jsonl)]
    C --> X[index_corpus<br/>embed]
    X --> V[(index.json<br/>vector store)]
    V --> E[eval_retrieval<br/>Hit@k, P@k, R@k, MRR]
    L[(retrieval_eval.json<br/>labelled queries)] --> E
  end
  subgraph API["FastAPI service"]
    S["POST /search"] --> Q[embed query] --> V
    CH["POST /chat"] --> O[Ollama /api/generate]
    H["GET /health"]
  end
```

## Features

- **Ingestion:** loads `.md`/`.txt`, chunks with overlap, and writes JSONL with deterministic chunk IDs (rebuilds are byte-for-byte reproducible).
- **Pluggable embedders** behind one `Embedder` interface:
  - `HashingEmbedder`: deterministic, offline and dependency-free, for development, tests and CI.
  - `OllamaEmbedder`: real embeddings from a local model (e.g. `nomic-embed-text`).
- **Vector store:** a teaching-sized store with cosine similarity, persisted as JSON. It does a brute-force scan, which is fine for thousands of chunks.
- **Retrieval evals:** labelled queries mapped to the chunks that answer them, with metrics as pure, unit-tested functions.
- **API:** `/health`, `/search` (top-k semantic search with scores and sources) and `/chat` (generation through Ollama, with typed errors mapped to 502/500).
- **Container:** a non-root Docker image that builds the sample index at image build time.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/health` | - | `{"status": "ok"}` |
| `POST` | `/search` | `{"query": "...", "top_k": 3}` (1-20) | `{"query", "hits": [{"chunk_id", "text", "score", "source"}]}`, or 503 if the index is empty |
| `POST` | `/chat` | `{"prompt": "...", "model": "mistral", "temperature": 0.2}` | `{"model", "response"}`, or 502 if Ollama is unreachable |

> `/chat` is plain generation today. Grounding chat answers in `/search` results is the next roadmap item.

## Quickstart

```sh
git clone https://github.com/rakshith-ponnappa/local-rag-agent-platform.git
cd local-rag-agent-platform
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Build the index from the sample docs (offline)
python -m agent_platform.rag_ingest --input-dir sample_docs --output-file artifacts/chunks.jsonl --chunk-size 400 --overlap 80
python -m agent_platform.index_corpus --chunks-file artifacts/chunks.jsonl --index-file artifacts/index.json --embedder hashing

# Run the API
uvicorn agent_platform.main:app --reload
```

```sh
curl -s localhost:8000/health
curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "how do pods get scheduled?", "top_k": 3}'

# /chat needs Ollama running locally:  ollama pull mistral
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"prompt": "Explain RAG in two sentences.", "model": "mistral"}'
```

With Docker:

```sh
docker build -t local-rag .
docker run --rm -p 8000:8000 local-rag     # reaches Ollama on the host via host.docker.internal
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint for `/chat` and `OllamaEmbedder` |
| `INDEX_FILE` | `artifacts/index.json` | Vector index loaded at startup |

## Evals

`evals/retrieval_eval.json` holds 14 queries over `sample_docs/`, each labelled with the chunk IDs
that answer it. `eval_retrieval` runs the retriever for every query and averages:

- **Hit@k:** did any relevant chunk appear in the top k?
- **Precision@k / Recall@k:** the share of the top k that's relevant, and the share of relevant chunks found.
- **MRR:** 1 / rank of the first relevant chunk.

```sh
python -m agent_platform.eval_retrieval --index-file artifacts/index.json --eval-file evals/retrieval_eval.json --k 3
```

Baseline with the offline `HashingEmbedder` (k = 3, chunk 400 / overlap 80):

| Hit@3 | Precision@3 | Recall@3 | MRR |
|---|---|---|---|
| 0.857 | 0.310 | 0.857 | 0.810 |

Precision is capped by the labels: most queries have a single relevant chunk, so a perfect top-3
scores 0.33. Use this baseline to compare embedders (`--embedder ollama`), chunk sizes or `top_k`.
CI re-runs the eval on every push.

## Testing

```sh
pytest
```

19 offline tests cover ingestion and chunking, the vector store, the metrics, eval loading and the
API (`/chat` with a mocked LLM, `/search` with a temporary index).

## Project layout

```text
src/agent_platform/
  main.py            FastAPI app: /health, /search, /chat
  rag_ingest.py      load + chunk documents -> JSONL
  index_corpus.py    embed chunks -> vector index
  embeddings.py      Embedder interface, HashingEmbedder, OllamaEmbedder
  vector_store.py    cosine-similarity store with JSON persistence
  eval_metrics.py    Hit@k, Precision@k, Recall@k, MRR
  eval_retrieval.py  run the labelled eval set
  llm_service.py     async Ollama client with typed errors
sample_docs/         small self-written corpus (Kubernetes, observability, vector search)
evals/               labelled retrieval queries
artifacts/           generated chunks and index for the sample corpus
```

## Design decisions

- **Evals before features.** A retrieval change is only an improvement if the numbers move, so the eval runs in CI from day one.
- **A deterministic embedder for CI.** Hashing embeddings make tests and evals reproducible without downloading models; real embeddings are a flag away.
- **Small, swappable parts.** The embedder, store and LLM sit behind tiny interfaces, so each can be replaced (pgvector, Qdrant, Azure OpenAI) without touching the API.
- **Typed failure modes.** An unreachable LLM is a 502 and an empty model response a 500, which keeps the service observable rather than silently returning empty answers.

## Roadmap

- [ ] RAG-grounded `/chat`: retrieve, build the prompt with citations, generate
- [ ] Reranking (cross-encoder) and hybrid BM25 + vector search
- [ ] OpenTelemetry traces and token/latency metrics per request
- [ ] Helm chart and Kubernetes deployment with an Ollama sidecar or a GPU node pool
- [ ] Expose `/search` as an MCP tool for agents

## License

[MIT](LICENSE)
