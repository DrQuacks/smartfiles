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
        # Explicitly use cosine distance for this collection so that
        # distances correspond to ``1 - cosine_similarity`` and the
        # scoring logic can safely treat them as such.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        self._client.delete_collection(name=self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

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

        # Always ask Chroma to return distances so we can log and
        # inspect them when needed.
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        # Optional diagnostic logging: when SMARTFILES_DEBUG_SCORES is
        # set, log raw distances and derived cosine similarities for
        # manual inspection.
        debug_scores = os.getenv("SMARTFILES_DEBUG_SCORES", "").lower() in {"1", "true", "yes"}

        for rank, (_id, doc, meta, dist) in enumerate(
            zip(ids, docs, metadatas, distances), start=1
        ):
            if dist is None:
                score = 0.0
                sim = 0.0
            else:
                raw_dist = float(dist)
                # Assuming Chroma is configured for cosine distance,
                # convert back to a cosine-like similarity.
                sim = 1.0 - raw_dist
                # Clamp to the theoretical cosine range [-1, 1].
                sim = max(-1.0, min(1.0, sim))
                score = (sim + 1.0) / 2.0 * 100.0

            if debug_scores:
                print(
                    f"[DEBUG] rank={rank} id={_id} dist={dist!r} "
                    f"sim_from_dist={sim:.6f} score={score:.2f}",
                    flush=True,
                )

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
