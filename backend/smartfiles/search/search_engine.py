from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from smartfiles.embeddings.embedding_model import get_default_embedding_model
from smartfiles.database.vector_store import get_default_vector_store


def run_search(*, query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not query.strip():
        return []

    profile = os.getenv("SMARTFILES_PROFILE_SEARCH", "").lower() in {"1", "true", "yes"}

    t0 = time.perf_counter()
    embedder = get_default_embedding_model()
    t1 = time.perf_counter()
    store = get_default_vector_store(recreate=False)
    t2 = time.perf_counter()

    embedding = embedder.embed_texts([query])[0]
    t3 = time.perf_counter()
    results = store.search(query_embedding=embedding, k=k)
    t4 = time.perf_counter()

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
