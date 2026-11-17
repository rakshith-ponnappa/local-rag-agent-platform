import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_platform.embeddings import HashingEmbedder
from agent_platform.llm_service import (
    EmptyModelResponseError,
    LLMServiceError,
    OllamaService,
)
from agent_platform.vector_store import LocalVectorStore


class ChatRequest(BaseModel):
    model: str = Field(default="mistral")
    prompt: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    model: str
    response: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


app = FastAPI(title="Agent Platform API", version="0.1.0")
llm_service = OllamaService()

# Retrieval components. The embedder must match the one used at indexing time.
embedder = HashingEmbedder()
_index_file = Path(os.getenv("INDEX_FILE", "artifacts/index.json"))
vector_store: LocalVectorStore | None = (
    LocalVectorStore.load(_index_file) if _index_file.exists() else None
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if vector_store is None or len(vector_store) == 0:
        raise HTTPException(
            status_code=503,
            detail="Vector index is empty. Run agent_platform.index_corpus first.",
        )

    query_embedding = embedder.embed_one(req.query)
    results = vector_store.query(query_embedding, top_k=req.top_k)

    hits = [
        SearchHit(
            chunk_id=result.chunk_id,
            text=result.text,
            score=result.score,
            source=str(result.metadata.get("source", "")),
        )
        for result in results
    ]
    return SearchResponse(query=req.query, hits=hits)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        text = await llm_service.generate(
            model=req.model,
            prompt=req.prompt,
            temperature=req.temperature,
        )
    except EmptyModelResponseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(model=req.model, response=text)
