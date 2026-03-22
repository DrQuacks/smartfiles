from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings

from smartfiles.config import get_data_dir
from smartfiles.ingestion.chunker import DocumentChunk

DEFAULT_DB_DIR = get_data_dir() / "database"
DEFAULT_COLLECTION_NAME = "documents"


class ChromaVectorStore:
    def __init__(self, db_path: Path, collection_name: str = DEFAULT_COLLECTION_NAME):
        db_path = db_path.expanduser().resolve()
        db_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(db_path), settings=Settings())
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def reset(self) -> None:
        self._client.delete_collection(name=self._collection.name)
        self._collection = self._client.get_or_create_collection(name=self._collection.name)

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch")

        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas: List[Dict[str, Any]] = []
        for c in chunks:
            meta: Dict[str, Any] = {
                "filepath": c.filepath,
                "chunk_index": c.chunk_index,
            }
            if getattr(c, "page_start", None) is not None:
                meta["page_start"] = c.page_start
            if getattr(c, "page_end", None) is not None:
                meta["page_end"] = c.page_end
            metadatas.append(meta)

        self._collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if not query_embedding:
            return []
        result = self._collection.query(query_embeddings=[query_embedding], n_results=k)
        hits: List[Dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for _id, doc, meta, dist in zip(ids, docs, metadatas, distances):
            # Chroma returns a distance value where smaller is better.
            # For cosine distance (the default), values are typically
            # in [0, 2]. We convert this to a human-friendly similarity
            # score in [0, 100], where higher is better.
            if dist is None:
                score = 0.0
            else:
                sim = 1.0 - float(dist)  # similarity ~ 1 - distance
                # Clamp to a reasonable cosine range [-1, 1].
                sim = max(-1.0, min(1.0, sim))
                score = (sim + 1.0) / 2.0 * 100.0
            item: Dict[str, Any] = {
                "id": _id,
                "text": doc,
                "score": score,
            }
            if isinstance(meta, dict):
                item.update(meta)
            hits.append(item)

        return hits


def get_default_vector_store(*, recreate: bool = False) -> ChromaVectorStore:
    store = ChromaVectorStore(db_path=DEFAULT_DB_DIR)
    if recreate:
        store.reset()
    return store
