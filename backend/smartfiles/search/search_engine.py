from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from smartfiles.embeddings.embedding_model import EmbeddingModel, get_default_embedding_model
from smartfiles.database.vector_store import ChromaVectorStore, get_default_vector_store


def run_search(
    *,
    query: str,
    k: int = 5,
    embedder: Optional[EmbeddingModel] = None,
    store: Optional[ChromaVectorStore] = None,
) -> List[Dict[str, Any]]:
    if not query.strip():
        return []

    profile = os.getenv("SMARTFILES_PROFILE_SEARCH", "").lower() in {"1", "true", "yes"}

    t0 = time.perf_counter()
    if embedder is None:
        embedder = get_default_embedding_model()
    t1 = time.perf_counter()
    if store is None:
        store = get_default_vector_store(recreate=False)
    t2 = time.perf_counter()

    embedding = embedder.embed_texts([query])[0]
    t3 = time.perf_counter()
    results = store.search(query_embedding=embedding, k=k)
    t4 = time.perf_counter()

    # Lightweight reranking: boost results where query terms appear
    # in the filename or chunk text. This keeps the primary vector
    # search but nudges obviously relevant documents upward, without
    # suppressing purely semantic matches (for example, a document
    # about lizards when searching for "gecko").
    tokens = [t.lower() for t in query.split() if len(t) > 2]
    if tokens and results:
        rescored: List[Dict[str, Any]] = []
        for item in results:
            score = float(item.get("score", 0.0))
            text = str(item.get("text", "") or "").lower()
            filepath = str(item.get("filepath", "") or "").lower()
            filename = filepath.rsplit("/", 1)[-1]

            filename_hits = sum(1 for tok in tokens if tok in filename)
            text_hits = sum(1 for tok in tokens if tok in text)

            # Cap positive contributions so long documents don't dominate.
            filename_boost = min(15.0, 5.0 * filename_hits)
            text_boost = min(10.0, 2.0 * text_hits)

            combined = score + filename_boost + text_boost

            # Store a transient combined score for sorting only.
            item["_combined_score"] = combined
            rescored.append(item)

        rescored.sort(
            key=lambda it: (
                -(it.get("_combined_score", it.get("score", 0.0)) or 0.0),
                -float(it.get("score", 0.0) or 0.0),
            )
        )

        for it in rescored:
            it.pop("_combined_score", None)

        results = rescored

    if profile:
        model_ms = (t1 - t0) * 1000.0
        store_ms = (t2 - t1) * 1000.0
        embed_ms = (t3 - t2) * 1000.0
        search_ms = (t4 - t3) * 1000.0
        total_ms = (t4 - t0) * 1000.0
        print(
            "[PROFILE] search "
            f"model_load={model_ms:.1f}ms store_init={store_ms:.1f}ms "
            f"embed={embed_ms:.1f}ms vector_search={search_ms:.1f}ms "
            f"total={total_ms:.1f}ms"
        )

    return results
