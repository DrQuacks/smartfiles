from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from smartfiles.embeddings.embedding_model import EmbeddingModel, get_default_embedding_model
from smartfiles.database.vector_store import ChromaVectorStore, get_default_vector_store
from smartfiles.ingestion.indexer import (
    extract_documents,
    build_index_from_corpus,
    run_indexing_pipeline,
    index_progress,
)
from smartfiles.search.search_engine import run_search
from smartfiles.search.dimdrop import add_dimdrop_similarity_scores, dimdrop_field_for_fraction
from smartfiles.search.reranker import rerank
from smartfiles.folder_registry import (
    FolderEntry,
    delete_folder_by_name,
    list_folders,
    reorder_folders,
)


app = FastAPI(title="SmartFiles API", version="0.1.0")


# In the local desktop app we want to be permissive so that the
# frontend (typically running on localhost:5173 via Vite) can reach
# the API without CORS issues.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    score_drop50: Optional[float] = None
    score_drop75: Optional[float] = None
    score_drop90: Optional[float] = None
    score_drop95: Optional[float] = None
    filepath: Optional[str] = None
    chunk_index: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    folder_name: Optional[str] = None


class RerankItem(BaseModel):
    id: str
    text: str
    score: float
    filepath: Optional[str] = None


class RerankRequest(BaseModel):
    query: str
    results: list[RerankItem]


class RerankResult(BaseModel):
    id: str
    rerank_score: float


class DimdropItem(BaseModel):
    id: str
    text: str
    score: float
    filepath: Optional[str] = None


class DimdropRequest(BaseModel):
    query: str
    results: list[DimdropItem]
    drop_fraction: float


class DimdropResult(BaseModel):
    id: str
    score: float


class FolderInfo(BaseModel):
    folder_name: str
    path: str
    raw_text_dir_name: str
    last_indexed: Optional[str] = None
    last_commit: Optional[str] = None


class ReorderFoldersRequest(BaseModel):
    order: list[str]


class IndexProgressResponse(BaseModel):
    stage: str
    current: int
    total: int


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

    # Reset and run a fresh indexing pipeline.
    index_progress.reset()
    run_indexing_pipeline(
        root_folder=root,
        recreate=payload.recreate,
        save_chunks=payload.save_chunks,
        chunk_size=payload.chunk_size,
        overlap=payload.overlap,
    )
    return {"status": "ok"}


@app.get("/index/progress", response_model=IndexProgressResponse)
def api_index_progress() -> IndexProgressResponse:
    """Return best-effort progress for the current indexing run.

    This is a simple in-memory tracker intended for single-user local
    use; it is reset at the start of each /index call.
    """

    return IndexProgressResponse(
        stage=index_progress.stage,
        current=index_progress.current,
        total=index_progress.total,
    )


def _folder_name_for_filepath(filepath: str, entries: List[FolderEntry]) -> Optional[str]:
    """Return the logical folder_name for a given file path, if any.

    This walks the registry entries and finds the longest root path
    that is a prefix of the given file path. If none match, returns
    None and the result is treated as unscoped.
    """

    if not filepath:
        return None

    file_path = Path(filepath).expanduser().resolve()

    best_name: Optional[str] = None
    best_len = -1
    for entry in entries:
        root = Path(entry.path).expanduser().resolve()
        try:
            file_path.relative_to(root)
        except ValueError:
            continue
        root_len = len(str(root))
        if root_len > best_len:
            best_len = root_len
            best_name = entry.folder_name

    return best_name


@app.get("/folders", response_model=list[FolderInfo])
def api_folders() -> list[FolderInfo]:
    """Return all folders that have been indexed.

    Only entries with a non-empty ``last_indexed`` field are returned,
    so the dropdown shows folders that have completed at least one
    extraction/indexing run. The ordering reflects the registry order,
    which can be adjusted via the reorder endpoint.
    """

    entries = list_folders()
    indexed: list[FolderInfo] = []
    for entry in entries:
        if not entry.last_indexed:
            continue
        indexed.append(
            FolderInfo(
                folder_name=entry.folder_name,
                path=entry.path,
                raw_text_dir_name=entry.raw_text_dir_name,
                last_indexed=entry.last_indexed,
                last_commit=entry.last_commit,
            )
        )
    return indexed


@app.delete("/folders/{folder_name}")
def api_delete_folder(folder_name: str) -> dict:
    """Delete a folder entry from the registry by its logical name.

    This affects ordering and visibility in the UI only; it does not
    currently delete any on-disk data or index contents.
    """

    removed = delete_folder_by_name(folder_name)
    if not removed:
        raise HTTPException(status_code=404, detail="Folder not found")
    return {"status": "ok"}


@app.post("/folders/reorder", response_model=list[FolderInfo])
def api_reorder_folders(payload: ReorderFoldersRequest) -> list[FolderInfo]:
    """Update the registry ordering for folders.

    The client sends an ordered list of folder_name values; any
    registered folders not mentioned keep their relative order and
    appear after the explicitly ordered entries.
    """

    reordered_entries = reorder_folders(payload.order)

    indexed: list[FolderInfo] = []
    for entry in reordered_entries:
        if not entry.last_indexed:
            continue
        indexed.append(
            FolderInfo(
                folder_name=entry.folder_name,
                path=entry.path,
                raw_text_dir_name=entry.raw_text_dir_name,
                last_indexed=entry.last_indexed,
                last_commit=entry.last_commit,
            )
        )
    return indexed


@app.get("/search", response_model=list[SearchResponse])
def api_search(query: str, k: int = 5, folders: Optional[str] = None) -> list[SearchResponse]:
    if not query.strip():
        return []

    if state.embedder is None or state.vector_store is None:
        raise HTTPException(status_code=500, detail="Embedding model or vector store not initialized")

    results = run_search(query=query, k=k, embedder=state.embedder, store=state.vector_store)

    # Attach folder_name to each result if we can resolve it from the
    # registry, and optionally filter to a subset of folders when the
    # client provides a comma-separated list of folder names.
    registry_entries = list_folders()
    allowed: Optional[set[str]] = None
    if folders:
        names = [name.strip() for name in folders.split(",") if name.strip()]
        if names:
            allowed = set(names)

    filtered: list[dict] = []
    for item in results:
        filepath = item.get("filepath") if isinstance(item, dict) else None
        folder_name = _folder_name_for_filepath(filepath or "", registry_entries)
        if folder_name:
            item["folder_name"] = folder_name

        if allowed is not None:
            # When a filter is provided, only include items that
            # belong to one of the requested folders.
            if folder_name in allowed:
                filtered.append(item)
        else:
            filtered.append(item)

    return [SearchResponse(**r) for r in filtered]


@app.post("/search/rerank", response_model=list[RerankResult])
def api_search_rerank(payload: RerankRequest) -> list[RerankResult]:
    """Re-rank an existing set of search results with a cross-encoder.

    The client first calls `/search` to obtain a list of candidates,
    then posts those candidates here along with the original query.
    This endpoint scores the candidates with a cross-encoder model
    and returns per-item `rerank_score` values sorted in descending
    order. The caller can merge these scores back into its own
    representation and optionally reorder the list.
    """

    if not payload.query.strip():
        return []

    items = [
        {
            "id": r.id,
            "text": r.text,
            "score": float(r.score),
            "filepath": r.filepath,
        }
        for r in payload.results
    ]

    scored = rerank(payload.query, items)
    return [RerankResult(id=it["id"], rerank_score=it["rerank_score"]) for it in scored]


@app.post("/search/dimdrop", response_model=list[DimdropResult])
def api_search_dimdrop(payload: DimdropRequest) -> list[DimdropResult]:
    """Compute dim-drop similarity scores for an already-ranked result list.

    This endpoint is intentionally separate from `/search` so we can:
    1) return initial rankings quickly, and
    2) progressively fill drop-score variants (20/40/60/80) in the UI.
    """

    if not payload.query.strip():
        return []

    if state.embedder is None:
        raise HTTPException(status_code=500, detail="Embedding model not initialized")

    field = dimdrop_field_for_fraction(payload.drop_fraction)
    if field is None:
        raise HTTPException(
            status_code=400,
            detail="drop_fraction must be one of 0.5, 0.75, 0.9, 0.95",
        )

    items = [
        {
            "id": r.id,
            "text": r.text,
            "score": float(r.score),
            "filepath": r.filepath,
        }
        for r in payload.results
    ]

    query_embedding = state.embedder.embed_texts([payload.query])[0]
    add_dimdrop_similarity_scores(
        embedder=state.embedder,
        query_embedding=query_embedding,
        results=items,
        drop_fractions=[payload.drop_fraction],
    )

    out: list[DimdropResult] = []
    for item in items:
        score = item.get(field)
        if score is None:
            continue
        out.append(DimdropResult(id=str(item.get("id", "")), score=float(score)))

    return out


@app.get("/file")
def api_file(filepath: str) -> FileResponse:
    """Serve the original file for a search result.

    This is intended for local use only and assumes that filepaths
    come from SmartFiles' own indexing process.
    """

    path = Path(filepath).expanduser()
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path)
