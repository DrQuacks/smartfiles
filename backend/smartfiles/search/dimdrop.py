from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from smartfiles.embeddings.embedding_model import EmbeddingModel


_DEBUG_DIMDROP = os.getenv("SMARTFILES_DEBUG_DIMDROP", "").lower() in {"1", "true", "yes"}

_DIMDROP_FIELD_BY_PERCENT: Dict[int, str] = {
    50: "score_drop50",
    75: "score_drop75",
    90: "score_drop90",
    95: "score_drop95",
}


def dimdrop_field_for_fraction(drop_fraction: float) -> str | None:
    """Return the response field name for a drop fraction.

    Examples:
    - 0.5 -> "score_drop50"
    - 0.75 -> "score_drop75"
    """

    pct = int(round(float(drop_fraction) * 100.0))
    return _DIMDROP_FIELD_BY_PERCENT.get(pct)


def _build_drop_masks(dim_order_asc: np.ndarray, dim: int, drop_fractions: Sequence[float]) -> Dict[float, np.ndarray]:
    """Build boolean masks for each drop fraction.

    Each mask has length ``dim``; entries set to ``False`` indicate
    dimensions that are dropped for that fraction.
    """

    masks: Dict[float, np.ndarray] = {}
    for frac in drop_fractions:
        if frac <= 0.0:
            mask = np.ones(dim, dtype=bool)
        elif frac >= 1.0:
            # Avoid dropping all dimensions; keep at least one with the
            # highest variance.
            mask = np.zeros(dim, dtype=bool)
            keep_idx = dim_order_asc[-1]
            mask[keep_idx] = True
        else:
            drop_count = int(dim * float(frac))
            drop_count = max(0, min(drop_count, dim - 1))
            mask = np.ones(dim, dtype=bool)
            mask[dim_order_asc[:drop_count]] = False
        masks[float(frac)] = mask
    return masks


def add_dimdrop_similarity_scores(
    *,
    embedder: EmbeddingModel,
    query_embedding: Sequence[float],
    results: List[Dict[str, Any]],
    drop_fractions: Iterable[float] = (0.5, 0.75, 0.9, 0.95),
) -> None:
    """Augment results with similarity scores under dim-drop variants.

    This keeps the input ``results`` list order unchanged and simply
    attaches additional keys on each item:

    - ``score_drop50`` – 50% lowest-variance dimensions removed
    - ``score_drop75`` – 75% lowest-variance dimensions removed
    - ``score_drop90`` – 90% lowest-variance dimensions removed
    - ``score_drop95`` – 95% lowest-variance dimensions removed

    Scores are computed as cosine similarities mapped to the same
    0–100 scale used by ``ChromaVectorStore.search``.
    """

    if not results:
        return

    # Resolve requested fractions to known output field names.
    drop_fracs = [float(f) for f in drop_fractions]
    fraction_to_field: Dict[float, str] = {}
    for frac in drop_fracs:
        field = dimdrop_field_for_fraction(frac)
        if field is not None:
            fraction_to_field[frac] = field
    if not fraction_to_field:
        return

    # Normalize the query embedding.
    q = np.asarray(list(query_embedding), dtype=np.float32)
    if q.ndim != 1:
        return

    # Re-embed the retrieved chunk texts with the same model. This
    # ensures we have embeddings even if Chroma is configured not to
    # return or persist them for diagnostics.
    texts: List[str] = [str(item.get("text", "")) for item in results]
    try:
        doc_vectors = embedder.embed_texts(texts)
    except Exception:
        # If embedding fails for any reason, bail out quietly; this is
        # an experimental diagnostics path and should not break search.
        return

    docs = np.asarray(doc_vectors, dtype=np.float32)
    if docs.ndim != 2 or docs.shape[0] == 0:
        return

    dim = docs.shape[1]
    if q.shape[0] != dim:
        return

    std = docs.std(axis=0)
    var = std ** 2

    # Dimensions ordered by increasing variance (lowest first) within
    # this retrieved set.
    dim_order_asc = np.argsort(var)
    masks = _build_drop_masks(dim_order_asc, dim=dim, drop_fractions=drop_fracs)

    # Precompute masked query vectors and norms for each fraction.
    q_masked: Dict[float, Tuple[np.ndarray, float]] = {}
    for frac, mask in masks.items():
        q_m = q[mask]
        norm_q = float(np.linalg.norm(q_m))
        q_masked[frac] = (q_m, norm_q)

    # Finally, compute per-result scores under each dim-drop scheme,
    # keeping the original ranking order intact.
    debug_samples: List[Dict[str, Any]] = []
    for item, v in zip(results, docs):
        _id = item.get("id")
        if not isinstance(_id, str):
            continue

        for frac in drop_fracs:
            field = fraction_to_field.get(frac)
            if not field:
                continue

            mask = masks[frac]
            q_m, norm_q = q_masked[frac]
            v_m = v[mask]
            norm_v = float(np.linalg.norm(v_m))

            if norm_q == 0.0 or norm_v == 0.0:
                sim = 0.0
            else:
                sim = float(np.dot(q_m, v_m) / (norm_q * norm_v))
                sim = max(-1.0, min(1.0, sim))

            # Map cosine similarity in [-1, 1] to [0, 100].
            score = (sim + 1.0) / 2.0 * 100.0
            item[field] = score

        if _DEBUG_DIMDROP and len(debug_samples) < 5:
            debug_samples.append(
                {
                    "id": _id,
                    "base": float(item.get("score", 0.0)),
                    "drop50": float(item.get("score_drop50", float("nan"))),
                    "drop75": float(item.get("score_drop75", float("nan"))),
                    "drop90": float(item.get("score_drop90", float("nan"))),
                    "drop95": float(item.get("score_drop95", float("nan"))),
                }
            )

    if _DEBUG_DIMDROP and debug_samples:
        print("[DIMDROP] sample scores:")
        for row in debug_samples:
            print("  ", row)
