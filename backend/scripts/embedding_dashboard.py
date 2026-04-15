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
    embeddings = base.get("embeddings")
    try:
        has_embeddings = embeddings is not None and len(embeddings) > 0
    except TypeError:
        # Fallback if object doesn't support len().
        has_embeddings = bool(embeddings is not None)
    if has_embeddings:
        return base

    # Otherwise, try to fetch full records for the peeked IDs.
    ids = base.get("ids") or []
    # Normalize possible NumPy arrays to lists to avoid ambiguous
    # truth-value checks.
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        # ``peek`` may return ids as ``[[...]]`` similar to ``query``.
        ids = ids[0]

    if isinstance(ids, (list, tuple)) and len(ids) > 0:
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

    embeddings = peek_result.get("embeddings")
    # Normalize possible NumPy arrays or None into a plain sequence
    # before checking emptiness.
    if isinstance(embeddings, np.ndarray):
        if embeddings.size == 0:
            embeddings = []
    elif embeddings is None:
        embeddings = []
    metadatas = peek_result.get("metadatas") or []

    try:
        is_empty = len(embeddings) == 0
    except TypeError:
        is_empty = False

    if is_empty:
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

    # Per-dimension mean, standard deviation, and range.
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    dim_min = embeddings.min(axis=0)
    dim_max = embeddings.max(axis=0)

    # Sort dimensions by variance (descending) for inspection.
    var = std ** 2
    order = np.argsort(var)[::-1]

    stats: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "dim": int(dim),
        "norms": norms,
        "mean": mean,
        "std": std,
        "min": dim_min,
        "max": dim_max,
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


def toggle_help(current: str, section: str) -> str:
    """Pure helper for help toggle state.

    Returns the new active section name given the current one and the
    section that was clicked. Clicking the same section twice clears it;
    clicking a different one switches focus.
    """

    return "" if current == section else section


def main() -> None:
    st.set_page_config(page_title="SmartFiles Embedding Explorer", layout="wide")
    st.title("SmartFiles Embedding Explorer")

    data_dir = get_data_dir()
    db_dir = DEFAULT_DB_DIR

    st.caption(f"Data directory: {data_dir}")
    st.caption(f"Chroma DB path: {db_dir}")

    # Initialize a single active help section identifier so that at
    # most one tooltip is open at a time.
    if "active_help_section" not in st.session_state:
        st.session_state["active_help_section"] = ""

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

        # Inline tooltip toggle for the sampling section. Always shows
        # a single `[?]` button and uses the sidebar text to indicate
        # which help section is open.
        row = st.columns([4, 1])
        with row[0]:
            st.caption("What does sampling control?")
        with row[1]:
            if st.button("[?]", key="btn_help_sampling"):
                st.session_state["active_help_section"] = toggle_help(
                    st.session_state.get("active_help_section", ""), "sampling"
                )
        if st.session_state.get("active_help_section", "") == "sampling":
            st.info(
                "Sampling controls how many embeddings we pull from the index "
                "for analysis. Larger samples give more stable statistics but "
                "take longer to compute. This does not change your SmartFiles "
                "index; it only affects this dashboard's estimates."
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

    # Header with inline help toggle for norm distribution.
    hdr_norms = st.columns([4, 1])
    with hdr_norms[0]:
        st.subheader("Vector norm distribution")
    with hdr_norms[1]:
        if st.button("[?]", key="btn_help_norms"):
            st.session_state["active_help_section"] = toggle_help(
                st.session_state.get("active_help_section", ""), "norms"
            )
    # Toggle lives next to the header; explanatory text is rendered in
    # the sidebar's "Section help" area.

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

    # Header with inline help toggle for per-dimension stats.
    hdr_dims = st.columns([4, 3, 1])
    with hdr_dims[0]:
        st.subheader("Per-dimension statistics")
    with hdr_dims[1]:
        # Sorting controls for the per-dimension table.
        sort_col = st.selectbox(
            "Sort by",
            options=["variance", "std", "mean", "range", "dim"],
            index=0,
            key="dims_sort_col",
        )
        sort_dir = st.radio(
            "Direction",
            options=["descending", "ascending"],
            index=0,
            horizontal=True,
            key="dims_sort_dir",
        )
    with hdr_dims[2]:
        if st.button("[?]", key="btn_help_dims"):
            st.session_state["active_help_section"] = toggle_help(
                st.session_state.get("active_help_section", ""), "dims"
            )
    # Toggle lives next to the header; explanatory text is rendered in
    # the sidebar's "Section help" area.

    var: np.ndarray = stats["var"]
    mean: np.ndarray = stats["mean"]
    std: np.ndarray = stats["std"]
    dim_min: np.ndarray = stats["min"]
    dim_max: np.ndarray = stats["max"]

    # Build a full per-dimension table (one row per embedding dimension).
    dim_indices = np.arange(len(var))
    dim_range = dim_max - dim_min
    df_dims = pd.DataFrame(
        {
            "dim": dim_indices,
            "variance": var,
            "std": std,
            "mean": mean,
            "min": dim_min,
            "max": dim_max,
            "range": dim_range,
        }
    )

    ascending = sort_dir == "ascending"
    df_dims_sorted = df_dims.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)

    st.dataframe(df_dims_sorted, use_container_width=True)

    # Header with inline help toggle for PCA.
    hdr_pca = st.columns([4, 1])
    with hdr_pca[0]:
        st.subheader("PCA projection (2D)")
    with hdr_pca[1]:
        if st.button("[?]", key="btn_help_pca"):
            st.session_state["active_help_section"] = toggle_help(
                st.session_state.get("active_help_section", ""), "pca"
            )
    # Toggle lives next to the header; explanatory text is rendered in
    # the sidebar's "Section help" area.

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

    # Render section help and debug state *after* all buttons have had
    # a chance to update active_help_section in this run.
    with st.sidebar:
        st.markdown("---")
        st.header("Section help")

        active = st.session_state.get("active_help_section", "")

        if active == "norms":
            st.subheader("Vector norm distribution")
            st.info(
                "This histogram shows the L2 length (norm) of each sampled "
                "embedding. If almost all norms cluster around a single value, "
                "it suggests the encoder (or cosine normalization) is keeping "
                "vectors at nearly constant length. A wider spread means some "
                "documents get much larger or smaller magnitudes than others."
            )

        if active == "dims":
            st.subheader("Per-dimension statistics (top-variance dimensions)")
            st.info(
                "The table lists the embedding dimensions with the largest "
                "variance across your sampled documents. High-variance "
                "dimensions are where the encoder is spreading points out; "
                "low-variance ones behave more like a shared background. "
                "Means near zero suggest a roughly centered dimension; large "
                "means indicate a persistent bias in that direction."
            )

        if active == "pca":
            st.subheader("PCA projection (2D)")
            st.info(
                "PCA finds the directions of greatest variance in the "
                "sampled embeddings and projects them into 2D for plotting. "
                "Each point is a document chunk; proximity in this plot "
                "reflects similarity along those top components only. The "
                "explained variance ratios tell you how much of the total "
                "spread is captured by PC1 and PC2."
            )

        with st.expander("Debug (tooltip state)", expanded=False):
            st.code(
                f"active_help_section = {st.session_state.get('active_help_section', '')!r}",
                language="python",
            )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
