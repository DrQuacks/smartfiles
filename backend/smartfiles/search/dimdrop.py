from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from smartfiles.embeddings.embedding_model import EmbeddingModel


_DEBUG_DIMDROP = os.getenv("SMARTFILES_DEBUG_DIMDROP", "").lower() in {"1", "true", "yes"}


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
    drop_fractions: Iterable[float] = (0.2, 0.4, 0.6, 0.8),
) -> None:
    """Augment results with similarity scores under dim-drop variants.

    This keeps the input ``results`` list order unchanged and simply
    attaches additional keys on each item:

    - ``score_drop20`` – 20% lowest-variance dimensions removed
    - ``score_drop40`` – 40% lowest-variance dimensions removed
    - ``score_drop60`` – 60% lowest-variance dimensions removed
    - ``score_drop80`` – 80% lowest-variance dimensions removed

    Scores are computed as cosine similarities mapped to the same
    0–100 scale used by ``ChromaVectorStore.search``.
    """

    if not results:
        return

    # Map from fraction to response field suffix.
    fraction_to_field: Dict[float, str] = {
        0.2: "score_drop20",
        0.4: "score_drop40",
        0.6: "score_drop60",
        0.8: "score_drop80",
    }

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
    drop_fracs = [float(f) for f in drop_fractions]
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
                    "drop20": float(item.get("score_drop20", float("nan"))),
                    "drop40": float(item.get("score_drop40", float("nan"))),
                    "drop60": float(item.get("score_drop60", float("nan"))),
                    "drop80": float(item.get("score_drop80", float("nan"))),
                }
            )

    if _DEBUG_DIMDROP and debug_samples:
        print("[DIMDROP] sample scores:")
        for row in debug_samples:
            print("  ", row)
