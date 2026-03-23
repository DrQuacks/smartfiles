from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from smartfiles.embeddings.embedding_model import EmbeddingModel, get_default_embedding_model
from smartfiles.database.vector_store import ChromaVectorStore, get_default_vector_store
from smartfiles.ingestion.indexer import (
    extract_documents,
    build_index_from_corpus,
    run_indexing_pipeline,
)
from smartfiles.search.search_engine import run_search


app = FastAPI(title="SmartFiles API", version="0.1.0")


origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    embedder: Optional[EmbeddingModel] = None
    vector_store: Optional[ChromaVectorStore] = None


state = AppState()


@app.on_event("startup")
def startup_event() -> None:
    """Initialize shared embedding model and vector store once.

    This avoids paying model-load and DB-init costs on every request.
    """

    state.embedder = get_default_embedding_model()
    state.vector_store = get_default_vector_store(recreate=False)


class ExtractRequest(BaseModel):
    root_folder: str
    recreate_text: bool = False


class IndexFromTextRequest(BaseModel):
    root_folder: str
    recreate: bool = False
    save_chunks: bool = True
    chunk_size: int = 500
    overlap: int = 50


class IndexRequest(BaseModel):
    root_folder: str
    recreate: bool = False
    save_chunks: bool = True
    chunk_size: int = 500
    overlap: int = 50


class SearchResponse(BaseModel):
    id: str
    text: str
    score: float
    filepath: Optional[str] = None
    chunk_index: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@app.get("/health")
def api_health() -> dict:
    """Simple health check endpoint.

    Returns basic status and whether core components are initialized.
    """

    return {
        "status": "ok",
        "model_loaded": state.embedder is not None,
        "vector_store_initialized": state.vector_store is not None,
    }


@app.post("/extract")
def api_extract(payload: ExtractRequest) -> dict:
    root = Path(payload.root_folder).expanduser()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="root_folder must be an existing directory")

    extract_documents(root_folder=root, recreate_text=payload.recreate_text)
    return {"status": "ok"}


@app.post("/index-from-text")
def api_index_from_text(payload: IndexFromTextRequest) -> dict:
    root = Path(payload.root_folder).expanduser()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="root_folder must be an existing directory")

    build_index_from_corpus(
        root_folder=root,
        recreate_index=payload.recreate,
        save_chunks=payload.save_chunks,
        chunk_size=payload.chunk_size,
        overlap=payload.overlap,
    )
    return {"status": "ok"}


@app.post("/index")
def api_index(payload: IndexRequest) -> dict:
    root = Path(payload.root_folder).expanduser()
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=400, detail="root_folder must be an existing directory")

    run_indexing_pipeline(
        root_folder=root,
        recreate=payload.recreate,
        save_chunks=payload.save_chunks,
        chunk_size=payload.chunk_size,
        overlap=payload.overlap,
    )
    return {"status": "ok"}


@app.get("/search", response_model=list[SearchResponse])
def api_search(query: str, k: int = 5) -> list[SearchResponse]:
    if not query.strip():
        return []

    if state.embedder is None or state.vector_store is None:
        raise HTTPException(status_code=500, detail="Embedding model or vector store not initialized")

    results = run_search(query=query, k=k, embedder=state.embedder, store=state.vector_store)

    return [SearchResponse(**r) for r in results]
