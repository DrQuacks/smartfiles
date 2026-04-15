from __future__ import annotations

import math
import os
from typing import Iterable, List, Dict, Any

try:  # pragma: no cover - optional dependency handling
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - defensive
    CrossEncoder = None  # type: ignore


_reranker_model = None


def _get_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model  # type: ignore[return-value]

    if CrossEncoder is None:
        raise RuntimeError(
            "sentence-transformers is required for reranking but is not installed."
        )

    model_name = os.getenv(
        "SMARTFILES_RERANK_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    _reranker_model = CrossEncoder(model_name)
    return _reranker_model


def rerank(query: str, items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score and sort candidate items for a query using a cross-encoder.

    Each item should at least provide an ``id`` and ``text`` field. The
    returned list has the same items, augmented with a ``rerank_score``
    field and sorted in descending score order.
    """

    items_list = list(items)
    if not items_list:
        return []

    model = _get_model()
    pairs = [(query, str(it.get("text", ""))) for it in items_list]
    scores = [float(s) for s in model.predict(pairs)]

    # Query-relative normalization: spread scores across [0, 100]
    # based on the best and worst candidate for this query. This
    # preserves ordering while making the display values easier to
    # interpret than raw logits, and avoids collapsing many items to 0.
    s_min = min(scores)
    s_max = max(scores)
    if s_max == s_min:
        norm = [50.0 for _ in scores]
    else:
        norm = [((s - s_min) / (s_max - s_min)) * 100.0 for s in scores]

    for it, score in zip(items_list, norm):
        it["rerank_score"] = float(score)

    items_list.sort(key=lambda it: it.get("rerank_score", 0.0), reverse=True)
    return items_list
