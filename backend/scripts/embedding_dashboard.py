from __future__ import annotations

"""Streamlit dashboard for exploring SmartFiles embeddings.

Usage (from backend/):

    source .venv/bin/activate
    pip install .[benchmark]
    streamlit run scripts/embedding_dashboard.py

This connects directly to the SmartFiles Chroma database in
SMARTFILES_DATA_DIR/database, samples a subset of document chunk
embeddings, and provides basic geometric diagnostics such as:

- overall embedding dimensionality and sample size
- distribution of vector norms
- per-dimension mean and standard deviation
- a simple 2D PCA projection of the sampled embeddings

The goal is to help reason about how the current embedding model is
behaving on your actual indexed corpus, in the same spirit as the BEIR
benchmark dashboard but focused on the embedding space itself.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from chromadb.config import Settings

from smartfiles.config import get_data_dir
from smartfiles.database.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_DB_DIR


def get_collection() -> Tuple[chromadb.ClientAPI, Any]:
    """Return a Chroma client and the SmartFiles documents collection.

    This mirrors the configuration used by ``ChromaVectorStore`` so we
    are looking at the exact same index that the app uses.
    """

    db_path = DEFAULT_DB_DIR.expanduser().resolve()
    client = chromadb.PersistentClient(path=str(db_path), settings=Settings())
    collection = client.get_collection(name=DEFAULT_COLLECTION_NAME)
    return client, collection


def peek_embeddings(collection: Any, limit: int) -> Dict[str, Any]:
    """Peek at up to ``limit`` items from the collection, with embeddings.

    We rely on Chroma's ``peek`` API, which returns a small sample of
    items without requiring us to know their IDs in advance. Older
    versions may not support an ``include`` argument here, so we call
    ``peek`` in its simplest form and, if needed, follow up with
    ``get`` to fetch embeddings and metadata.
    """

    base = collection.peek(limit=limit)

    # If embeddings are already present (newer Chroma), just return.
    if base.get("embeddings"):
        return base

    # Otherwise, try to fetch full records for the peeked IDs.
    ids = base.get("ids") or []
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        # ``peek`` may return ids as ``[[...]]`` similar to ``query``.
        ids = ids[0]

    if ids:
        detail = collection.get(ids=ids, include=["embeddings", "documents", "metadatas"])
        return {
            "embeddings": detail.get("embeddings") or [],
            "documents": detail.get("documents") or [],
            "metadatas": detail.get("metadatas") or [],
        }

    return base


def extract_embedding_matrix(peek_result: Dict[str, Any]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Extract an (n_samples, dim) matrix and associated metadata list.

    Returns (embeddings, metadatas).
    """

    embeddings = peek_result.get("embeddings") or []
    metadatas = peek_result.get("metadatas") or []

    if not embeddings:
        raise RuntimeError(
            "No embeddings returned from Chroma. Make sure your index is "
            "built with embeddings stored, and that the collection is not empty."
        )

    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected embeddings shape: {arr.shape!r}")

    # Ensure metadatas length matches rows; if not, pad with empty dicts.
    if len(metadatas) < arr.shape[0]:
        metadatas = list(metadatas) + [{} for _ in range(arr.shape[0] - len(metadatas))]
    elif len(metadatas) > arr.shape[0]:
        metadatas = metadatas[: arr.shape[0]]

    return arr, metadatas


def compute_basic_stats(embeddings: np.ndarray) -> Dict[str, Any]:
    """Compute simple geometric diagnostics for a set of embeddings."""

    n_samples, dim = embeddings.shape

    # L2 norms per vector.
    norms = np.linalg.norm(embeddings, axis=1)

    # Per-dimension mean and standard deviation.
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)

    # Sort dimensions by variance (descending) for inspection.
    var = std ** 2
    order = np.argsort(var)[::-1]

    stats: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "dim": int(dim),
        "norms": norms,
        "mean": mean,
        "std": std,
        "var": var,
        "sorted_dims": order,
    }
    return stats


def compute_pca(embeddings: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a simple PCA projection using NumPy only.

    Returns (projected, explained_variance_ratio).
    """

    n_samples, dim = embeddings.shape
    if n_components <= 0 or n_components > dim:
        raise ValueError("n_components must be between 1 and embedding dimension")

    # Center the data.
    X = embeddings - embeddings.mean(axis=0, keepdims=True)

    # Compute SVD of the centered matrix. This is equivalent to PCA
    # on the covariance matrix but uses NumPy only (no sklearn).
    # X = U S V^T, where rows of V^T are principal directions.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Principal components (directions) are rows of Vt.
    components = Vt[:n_components, :]

    # Project data onto the top components.
    projected = X @ components.T

    # Explained variance for each component is S^2 / (n_samples - 1).
    if n_samples > 1:
        variances = (S ** 2) / float(n_samples - 1)
        total_var = variances.sum()
        explained = variances[:n_components] / total_var if total_var > 0 else np.zeros(n_components)
    else:
        explained = np.zeros(n_components)

    return projected, explained


def main() -> None:
    st.set_page_config(page_title="SmartFiles Embedding Explorer", layout="wide")
    st.title("SmartFiles Embedding Explorer")

    data_dir = get_data_dir()
    db_dir = DEFAULT_DB_DIR

    st.caption(f"Data directory: {data_dir}")
    st.caption(f"Chroma DB path: {db_dir}")

    try:
        client, collection = get_collection()
    except Exception as exc:  # pragma: no cover - UI-only
        st.error(f"Failed to open Chroma collection: {exc}")
        return

    # High-level info.
    try:
        total_count = collection.count()
    except Exception:
        total_count = None

    with st.sidebar:
        st.header("Sampling")
        max_default = 1000
        max_limit = 5000
        sample_limit = st.slider(
            "Number of embeddings to sample",
            min_value=100,
            max_value=max_limit,
            value=min(max_default, max_limit),
            step=100,
        )

        st.caption(
            "Embeddings are sampled using Chroma's `peek` API. "
            "This is not a truly random sample but is sufficient "
            "for geometric diagnostics."
        )

    cols = st.columns(3)
    with cols[0]:
        st.metric("Total indexed vectors", value=str(total_count) if total_count is not None else "unknown")
    with cols[1]:
        st.metric("Sample size", value=str(sample_limit))

    # Fetch sample.
    try:
        peek_result = peek_embeddings(collection, limit=sample_limit)
        embeddings, metadatas = extract_embedding_matrix(peek_result)
    except Exception as exc:  # pragma: no cover - UI-only
        st.error(f"Failed to load embeddings from Chroma: {exc}")
        return

    stats = compute_basic_stats(embeddings)
    n_samples = stats["n_samples"]
    dim = stats["dim"]

    with cols[2]:
        st.metric("Embedding dimension", value=str(dim))

    st.subheader("Vector norm distribution")

    norms: np.ndarray = stats["norms"]
    # Histogram for norms.
    if n_samples > 0:
        num_bins = int(math.sqrt(n_samples)) if n_samples > 0 else 10
        num_bins = max(10, min(60, num_bins))
        counts, bin_edges = np.histogram(norms, bins=num_bins)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        df_norms = pd.DataFrame({"norm": centers, "count": counts})
        st.bar_chart(df_norms, x="norm", y="count")

        st.caption(
            f"Norms: min={norms.min():.4f}, max={norms.max():.4f}, "
            f"mean={norms.mean():.4f}, std={norms.std():.4f}"
        )
    else:
        st.info("No embeddings available to compute norms.")

    st.subheader("Per-dimension statistics (top-variance dimensions)")

    var: np.ndarray = stats["var"]
    mean: np.ndarray = stats["mean"]
    std: np.ndarray = stats["std"]
    order: np.ndarray = stats["sorted_dims"]

    top_k = 32
    top_dims = order[:top_k]
    df_dims = pd.DataFrame(
        {
            "dim": top_dims,
            "variance": var[top_dims],
            "std": std[top_dims],
            "mean": mean[top_dims],
        }
    )
    st.dataframe(df_dims, use_container_width=True)

    st.subheader("PCA projection (2D)")

    try:
        projected, explained = compute_pca(embeddings, n_components=2)
    except Exception as exc:  # pragma: no cover - UI-only
        st.error(f"Failed to compute PCA: {exc}")
        return

    pc1 = projected[:, 0]
    pc2 = projected[:, 1]

    # Attach simple metadata for visualization.
    filepaths: List[str] = []
    folders: List[str] = []
    for meta in metadatas:
        if not isinstance(meta, dict):
            meta = {}
        path_str = str(meta.get("filepath") or "")
        filepaths.append(path_str)
        if path_str:
            p = Path(path_str)
            folders.append(p.parent.name)
        else:
            folders.append("")

    df_pca = pd.DataFrame(
        {
            "pc1": pc1,
            "pc2": pc2,
            "filepath": filepaths,
            "folder": folders,
        }
    )

    st.caption(
        "PCA is computed on the sampled embeddings only, using a "
        "pure NumPy SVD. Explained variance ratios for PC1/PC2: "
        f"{explained[0]:.4f}, {explained[1]:.4f}."
    )

    color_by = st.selectbox("Color points by", ["folder", "none"], index=0)

    if color_by == "folder":
        st.scatter_chart(df_pca, x="pc1", y="pc2", color="folder")
    else:
        st.scatter_chart(df_pca, x="pc1", y="pc2")

    with st.expander("Raw PCA sample (first 200 rows)"):
        st.dataframe(df_pca.head(200), use_container_width=True)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
