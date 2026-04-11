from __future__ import annotations

from pathlib import Path
from typing import Mapping, List
import os

from smartfiles.config import get_data_dir
from smartfiles.database.vector_store import ChromaVectorStore
from smartfiles.embeddings.embedding_model import (
    EmbeddingModel,
    get_default_embedding_model,
    PROFILE_ENV_VAR,
    MODEL_ENV_VAR,
)
from smartfiles.ingestion.chunker import DocumentChunk
from smartfiles.search.search_engine import run_search

from .base import RetrievalBackend
from ..core.models import Hit


class SmartFilesBackend(RetrievalBackend):
    """RetrievalBackend that uses SmartFiles' embedding + Chroma stack.

    This adapter mirrors the logic in `smartfiles.benchmarks.beir_runner`,
    but conforms to the generic RetrievalBackend interface used by the
    evaluator.
    """

    def __init__(self, collection_name: str = "beir-eval", batch_size: int = 128) -> None:
        self._collection_name = collection_name
        self._batch_size = batch_size
        self._store: ChromaVectorStore | None = None
        self._embedder: EmbeddingModel | None = None
        # Capture embedding-related environment so runs always carry
        # basic configuration metadata, even when invoked via the CLI.
        self.metadata = {
            "embedding_profile": os.getenv(PROFILE_ENV_VAR) or None,
            "embedding_model_override": os.getenv(MODEL_ENV_VAR) or None,
        }

    @property
    def name(self) -> str:
        return f"smartfiles:{self._collection_name}"

    def _db_dir(self) -> Path:
        # Keep evaluator indices under the SmartFiles data dir but separate
        # from regular user indexes.
        root = get_data_dir() / "benchmarks" / "evaluator"
        return root / self._collection_name / "database"

    def index_corpus(self, corpus: Mapping[str, Mapping[str, str]]) -> None:
        # Initialize store and embedder.
        db_dir = self._db_dir()
        db_dir.parent.mkdir(parents=True, exist_ok=True)
        self._store = ChromaVectorStore(db_path=db_dir, collection_name=self._collection_name)
        self._store.reset()
        self._embedder = get_default_embedding_model()

        # Prepare and index documents in batches to avoid high memory usage.
        doc_items = list(corpus.items())
        texts_batch: List[str] = []
        chunks_batch: List[DocumentChunk] = []

        def _flush_batch() -> None:
            if not chunks_batch or self._store is None or self._embedder is None:
                return
            embeddings = self._embedder.embed_texts(texts_batch)
            self._store.add_documents(chunks_batch, embeddings)
            texts_batch.clear()
            chunks_batch.clear()

        for doc_id, fields in doc_items:
            title = (fields.get("title") or "").strip()
            body = (fields.get("text") or "").strip()
            combined = (title + "\n" + body).strip() if title or body else ""
            if not combined:
                continue

            chunk = DocumentChunk(
                id=str(doc_id),
                filepath=str(doc_id),
                chunk_index=0,
                text=combined,
            )
            texts_batch.append(combined)
            chunks_batch.append(chunk)

            if len(texts_batch) >= self._batch_size:
                _flush_batch()

        _flush_batch()

    def search(self, query: str, top_k: int) -> List[Hit]:
        if self._store is None or self._embedder is None:
            raise RuntimeError("Store not built. Call index_corpus first.")

        hits_raw = run_search(query=query, k=top_k, embedder=self._embedder, store=self._store)
        hits: List[Hit] = []
        for item in hits_raw:
            # BEIR runner uses filepath or id as the BEIR doc id.
            doc_id = item.get("filepath") or item.get("id")
            if not doc_id:
                continue
            score = float(item.get("score", 0.0) or 0.0)
            hits.append(Hit(doc_id=str(doc_id), score=score))
        return hits
