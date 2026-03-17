from __future__ import annotations

from typing import Any, Dict, List

from smartfiles.embeddings.embedding_model import get_default_embedding_model
from smartfiles.database.vector_store import get_default_vector_store


def run_search(*, query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not query.strip():
        return []

    embedder = get_default_embedding_model()
    store = get_default_vector_store(recreate=False)

    embedding = embedder.embed_texts([query])[0]
    results = store.search(query_embedding=embedding, k=k)
    return results
