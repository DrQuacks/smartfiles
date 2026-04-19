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
import importlib
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from chromadb.config import Settings

from smartfiles.config import get_data_dir
from smartfiles.database.text_store import iter_corpus_documents
from smartfiles.database.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_DB_DIR
from smartfiles.embeddings.embedding_model import get_default_embedding_model
from smartfiles.folder_registry import list_folders


def get_collection(db_path: Path, collection_name: str) -> Tuple[chromadb.ClientAPI, Any]:
    """Return a Chroma client and the SmartFiles documents collection.

    This mirrors the configuration used by ``ChromaVectorStore`` so we
    are looking at the exact same index that the app uses.
    """

    db_path = db_path.expanduser().resolve()
    client = chromadb.PersistentClient(path=str(db_path), settings=Settings())
    collection = client.get_collection(name=collection_name)
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


def discover_embedding_sources(data_dir: Path, default_db_dir: Path, default_collection: str) -> List[Dict[str, str]]:
    """Discover preset embedding sources for the dashboard.

    Includes:
    - local SmartFiles index (documents collection)
    - BEIR dataset indexes under SMARTFILES_DATA_DIR/benchmarks/beir/*/database
    """

    sources: List[Dict[str, str]] = [
        {
            "label": "Local SmartFiles",
            "db_path": str(default_db_dir.expanduser().resolve()),
            "collection": default_collection,
        }
    ]

    beir_root = (data_dir / "benchmarks" / "beir").expanduser().resolve()
    if not beir_root.exists() or not beir_root.is_dir():
        return sources

    dataset_dirs = sorted([p for p in beir_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    for ds_dir in dataset_dirs:
        db_dir = ds_dir / "database"
        if not db_dir.exists() or not db_dir.is_dir():
            continue
        dataset = ds_dir.name
        sources.append(
            {
                "label": f"BEIR: {dataset}",
                "db_path": str(db_dir.expanduser().resolve()),
                "collection": f"beir-{dataset}",
            }
        )

    return sources


def _load_single_source_embeddings(
    db_dir: Path,
    collection_name: str,
    sample_limit: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], int | None]:
    """Load sampled embeddings for one collection."""

    _client, collection = get_collection(db_dir, collection_name)
    try:
        total_count = int(collection.count())
    except Exception:
        total_count = None

    peek_result = peek_embeddings(collection, limit=sample_limit)
    embeddings, metadatas = extract_embedding_matrix(peek_result)
    return embeddings, metadatas, total_count


def _load_beir_mix_embeddings(
    selected_sources: List[Dict[str, str]],
    sample_limit: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], int | None]:
    """Load and concatenate sampled embeddings from multiple BEIR collections."""

    if not selected_sources:
        raise RuntimeError("No BEIR datasets selected")

    per_source = max(1, sample_limit // len(selected_sources))
    remainder = sample_limit - (per_source * len(selected_sources))

    parts: List[np.ndarray] = []
    all_meta: List[Dict[str, Any]] = []
    total_count_sum = 0
    total_known = True

    target_dim: int | None = None
    for idx, src in enumerate(selected_sources):
        db_dir = Path(src["db_path"]).expanduser().resolve()
        collection_name = src["collection"]
        label = src["label"]

        this_limit = per_source + (1 if idx < remainder else 0)
        emb, meta, total_count = _load_single_source_embeddings(db_dir, collection_name, this_limit)

        if target_dim is None:
            target_dim = int(emb.shape[1])
        elif int(emb.shape[1]) != target_dim:
            raise RuntimeError(
                f"Dimension mismatch across selected sources: expected {target_dim}, got {emb.shape[1]} for {label}"
            )

        for m in meta:
            out = dict(m) if isinstance(m, dict) else {}
            out["source_label"] = label
            all_meta.append(out)

        parts.append(emb)
        if total_count is None:
            total_known = False
        else:
            total_count_sum += int(total_count)

    if not parts:
        raise RuntimeError("No embeddings loaded from selected BEIR datasets")

    stacked = np.vstack(parts)
    return stacked, all_meta, (total_count_sum if total_known else None)


def _sample_local_corpus_texts(root_folder: Path, max_texts: int, seed: int) -> List[str]:
    texts: List[str] = []
    for _path, text in iter_corpus_documents(root_folder):
        value = text.strip()
        if value:
            texts.append(value)

    if len(texts) <= max_texts:
        return texts

    rng = random.Random(seed)
    idx = rng.sample(range(len(texts)), k=max_texts)
    return [texts[i] for i in idx]


def _sample_beir_raw_texts(dataset: str, split: str, max_texts: int, seed: int) -> List[str]:
    try:
        from smartfiles.benchmarks.beir_runner import _download_and_load_beir
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "BEIR sampling requires benchmark dependencies. Run `pip install .[benchmark]`."
        ) from exc

    corpus, _queries, _qrels = _download_and_load_beir(dataset, split)

    texts: List[str] = []
    for fields in corpus.values():
        title = str((fields.get("title") or "")).strip()
        body = str((fields.get("text") or "")).strip()
        combined = (title + "\n" + body).strip() if title or body else ""
        if combined:
            texts.append(combined)

    if len(texts) <= max_texts:
        return texts

    rng = random.Random(seed)
    idx = rng.sample(range(len(texts)), k=max_texts)
    return [texts[i] for i in idx]


def _parse_hf_spec(spec: str) -> Tuple[str, str | None, str, str]:
    parts = [part.strip() for part in spec.split("::")]
    repo_id = parts[0] if parts else ""
    if not repo_id:
        raise ValueError(f"Invalid HF dataset spec: {spec!r}")

    config = parts[1] or None if len(parts) > 1 else None
    split = parts[2] or "train" if len(parts) > 2 else "train"
    text_field = parts[3] or "text" if len(parts) > 3 else "text"
    return repo_id, config, split, text_field


def _sample_hf_streaming_texts(
    spec: str,
    max_texts: int,
    seed: int,
    max_scan_examples: int,
) -> List[str]:
    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "HF streaming sampling requires `datasets`. Run `pip install .[benchmark]`."
        ) from exc

    repo_id, config, split, text_field = _parse_hf_spec(spec)
    ds = load_dataset(repo_id, name=config, split=split, streaming=True)

    rng = random.Random(seed)
    reservoir: List[str] = []
    seen_valid = 0

    for row_idx, row in enumerate(ds):
        if row_idx >= max_scan_examples:
            break
        value = row.get(text_field)
        text = str(value).strip() if value is not None else ""
        if not text:
            continue

        seen_valid += 1
        if len(reservoir) < max_texts:
            reservoir.append(text)
            continue

        replace_idx = rng.randint(0, seen_valid - 1)
        if replace_idx < max_texts:
            reservoir[replace_idx] = text

    return reservoir


def _load_raw_mix_embeddings(
    *,
    local_paths: List[Path],
    beir_datasets: List[str],
    hf_specs: List[str],
    beir_split: str,
    per_source_sample_size: int,
    hf_max_scan_examples: int,
    batch_size: int,
    seed: int,
    progress_callback=None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], int | None]:
    source_texts: List[Tuple[str, str]] = []
    source_cursor = 0

    total_sources = len(local_paths) + len(beir_datasets) + len(hf_specs)
    processed_sources = 0

    def _report_sampling(message: str) -> None:
        if progress_callback is not None:
            progress_callback("sampling", processed_sources, total_sources, message)

    for local_path in local_paths:
        texts = _sample_local_corpus_texts(local_path, per_source_sample_size, seed + source_cursor)
        source_cursor += 1
        processed_sources += 1
        source_label = f"local:{local_path.name}"
        source_texts.extend((text, source_label) for text in texts)
        _report_sampling(f"Sampled {len(texts)} texts from {source_label}")

    for dataset in beir_datasets:
        texts = _sample_beir_raw_texts(dataset, beir_split, per_source_sample_size, seed + source_cursor)
        source_cursor += 1
        processed_sources += 1
        source_label = f"beir:{dataset}"
        source_texts.extend((text, source_label) for text in texts)
        _report_sampling(f"Sampled {len(texts)} texts from {source_label}")

    for spec in hf_specs:
        texts = _sample_hf_streaming_texts(
            spec,
            max_texts=per_source_sample_size,
            seed=seed + source_cursor,
            max_scan_examples=hf_max_scan_examples,
        )
        source_cursor += 1
        processed_sources += 1
        source_label = f"hf:{spec}"
        source_texts.extend((text, source_label) for text in texts)
        _report_sampling(f"Sampled {len(texts)} texts from {source_label}")

    if not source_texts:
        raise RuntimeError("No raw sources selected")

    # Exact-text dedupe while preserving first source attribution.
    deduped: List[Tuple[str, str]] = []
    seen_texts = set()
    for text, src in source_texts:
        if text in seen_texts:
            continue
        seen_texts.add(text)
        deduped.append((text, src))

    if len(deduped) < 2:
        raise RuntimeError("Need at least two sampled texts to analyze embeddings")

    embedder = get_default_embedding_model()

    vectors: List[List[float]] = []
    metas: List[Dict[str, Any]] = []
    total_batches = max(1, math.ceil(len(deduped) / batch_size))
    for start in range(0, len(deduped), batch_size):
        batch = deduped[start : start + batch_size]
        batch_texts = [item[0] for item in batch]
        batch_src = [item[1] for item in batch]
        vectors.extend(embedder.embed_texts(batch_texts))
        for src in batch_src:
            metas.append({"filepath": "", "source_label": src})
        if progress_callback is not None:
            batch_idx = (start // batch_size) + 1
            progress_callback(
                "embedding",
                batch_idx,
                total_batches,
                f"Embedded batch {batch_idx}/{total_batches}",
            )

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected raw-mix embedding shape: {arr.shape!r}")
    return arr, metas, int(arr.shape[0])


def main() -> None:
    st.set_page_config(page_title="SmartFiles Embedding Explorer", layout="wide")
    st.title("SmartFiles Embedding Explorer")

    data_dir = get_data_dir()

    db_env = os.getenv("SMARTFILES_EMBEDDING_EXPLORER_DB_PATH", "").strip()
    col_env = os.getenv("SMARTFILES_EMBEDDING_EXPLORER_COLLECTION", "").strip()

    default_db_dir = Path(db_env).expanduser().resolve() if db_env else DEFAULT_DB_DIR
    default_collection = col_env or DEFAULT_COLLECTION_NAME

    preset_sources = discover_embedding_sources(data_dir, default_db_dir, default_collection)
    beir_sources = [s for s in preset_sources if s["label"].startswith("BEIR: ")]

    data_mode = "single"
    selected_beir_labels: List[str] = []
    selected_raw_local_labels: List[str] = []
    selected_raw_beir: List[str] = []
    selected_raw_hf: List[str] = []
    raw_beir_split = "test"
    raw_per_source_sample_size = 300
    raw_hf_max_scan_examples = 5000
    raw_embed_batch_size = 128
    raw_seed = 13
    db_dir = default_db_dir
    collection_name = default_collection

    registered_locals = list_folders()
    local_label_to_path: Dict[str, Path] = {
        f"{entry.folder_name} ({entry.path})": Path(entry.path).expanduser().resolve()
        for entry in registered_locals
    }
    beir_dataset_names = sorted([src["label"].replace("BEIR: ", "") for src in beir_sources])
    hf_default_specs = [
        "fancyzhx/ag_news",
        "google/wiki40b::en::train::text",
    ]

    with st.sidebar:
        st.header("Data Source")
        st.caption("Adjust settings, then click **Run Analysis** to refresh results.")
        data_mode = st.radio(
            "Source mode",
            options=["Single source", "BEIR mix", "Raw mixed sample (no index)"],
            index=0,
        )

        if data_mode == "Single source":
            selected_idx = st.selectbox(
                "Preset source",
                options=list(range(len(preset_sources))),
                format_func=lambda i: preset_sources[i]["label"],
                index=0,
            )

            preset = preset_sources[selected_idx]
            use_manual_override = st.checkbox("Manual override", value=False)

            if use_manual_override:
                db_dir_input = st.text_input("Chroma DB path", value=preset["db_path"])
                collection_name = st.text_input("Collection name", value=preset["collection"])
            else:
                db_dir_input = preset["db_path"]
                collection_name = preset["collection"]

            db_dir = Path(db_dir_input).expanduser().resolve()
        else:
            if data_mode == "BEIR mix":
                beir_labels = [s["label"] for s in beir_sources]
                selected_beir_labels = st.multiselect(
                    "BEIR datasets",
                    options=beir_labels,
                    default=beir_labels[: min(3, len(beir_labels))],
                )
                if not beir_sources:
                    st.caption("No BEIR datasets auto-detected yet. Build one to see it here.")
            else:
                st.caption(
                    "Raw mixed sample mode embeds sampled texts directly from local/BEIR/HF sources "
                    "without indexing into Chroma."
                )

                selected_raw_local_labels = st.multiselect(
                    "Local sources (registered folders)",
                    options=list(local_label_to_path.keys()),
                    default=list(local_label_to_path.keys())[:1],
                )

                selected_raw_beir = st.multiselect(
                    "BEIR raw datasets",
                    options=beir_dataset_names,
                    default=beir_dataset_names[: min(2, len(beir_dataset_names))],
                )

                selected_raw_hf = st.multiselect(
                    "HF streaming datasets",
                    options=hf_default_specs,
                    default=hf_default_specs,
                )

                raw_beir_split = st.selectbox("BEIR split", options=["test", "dev", "train"], index=0)
                raw_per_source_sample_size = st.slider(
                    "Raw sample size per source",
                    min_value=50,
                    max_value=2000,
                    value=300,
                    step=50,
                )
                raw_hf_max_scan_examples = st.slider(
                    "HF max scanned rows per source",
                    min_value=500,
                    max_value=50000,
                    value=5000,
                    step=500,
                )
                raw_embed_batch_size = st.slider(
                    "Embedding batch size (raw mode)",
                    min_value=16,
                    max_value=512,
                    value=128,
                    step=16,
                )
                raw_seed = st.number_input("Random seed", value=13, min_value=0, max_value=10_000)

        if len(preset_sources) == 1 and data_mode == "Single source":
            st.caption("No BEIR datasets auto-detected yet. Build one to see it here.")

    st.caption(f"Data directory: {data_dir}")
    if data_mode == "Single source":
        st.caption(f"Chroma DB path: {db_dir}")
        st.caption(f"Collection: {collection_name}")
    elif data_mode == "BEIR mix":
        selected_text = ", ".join(selected_beir_labels) if selected_beir_labels else "(none)"
        st.caption(f"BEIR mix sources: {selected_text}")
    else:
        local_text = ", ".join(selected_raw_local_labels) if selected_raw_local_labels else "(none)"
        beir_text = ", ".join(selected_raw_beir) if selected_raw_beir else "(none)"
        hf_text = ", ".join(selected_raw_hf) if selected_raw_hf else "(none)"
        st.caption(f"Raw local sources: {local_text}")
        st.caption(f"Raw BEIR sources: {beir_text}")
        st.caption(f"Raw HF sources: {hf_text}")

    # Initialize a single active help section identifier so that at
    # most one tooltip is open at a time.
    if "active_help_section" not in st.session_state:
        st.session_state["active_help_section"] = ""

    total_count: int | None = None

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

        if data_mode == "Raw mixed sample (no index)":
            st.caption(
                "In raw mixed mode, this limit is ignored. Use the raw-mode controls above "
                "to choose per-source sample size."
            )
        else:
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

    current_run_config: Dict[str, Any] = {
        "data_mode": data_mode,
        "db_dir": str(db_dir),
        "collection_name": collection_name,
        "sample_limit": int(sample_limit),
        "selected_beir_labels": selected_beir_labels,
        "selected_raw_local_labels": selected_raw_local_labels,
        "selected_raw_beir": selected_raw_beir,
        "selected_raw_hf": selected_raw_hf,
        "raw_beir_split": raw_beir_split,
        "raw_per_source_sample_size": int(raw_per_source_sample_size),
        "raw_hf_max_scan_examples": int(raw_hf_max_scan_examples),
        "raw_embed_batch_size": int(raw_embed_batch_size),
        "raw_seed": int(raw_seed),
    }

    with st.sidebar:
        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    has_previous_run = all(
        key in st.session_state
        for key in [
            "embedding_dashboard_last_run_config",
            "embedding_dashboard_embeddings",
            "embedding_dashboard_metadatas",
            "embedding_dashboard_total_count",
        ]
    )

    if run_clicked:
        progress = st.progress(0.0)
        progress_msg = st.empty()

        def _raw_progress(stage: str, current: int, total: int, message: str) -> None:
            safe_total = max(1, int(total))
            ratio = float(current) / float(safe_total)
            ratio = min(max(ratio, 0.0), 1.0)
            if stage == "sampling":
                overall = 0.5 * ratio
            elif stage == "embedding":
                overall = 0.5 + (0.5 * ratio)
            else:
                overall = ratio
            progress.progress(overall)
            progress_msg.info(message)

        try:
            if data_mode == "Single source":
                with st.spinner("Loading indexed embeddings from Chroma..."):
                    embeddings, metadatas, total_count = _load_single_source_embeddings(
                        db_dir=db_dir,
                        collection_name=collection_name,
                        sample_limit=sample_limit,
                    )
            elif data_mode == "BEIR mix":
                if not selected_beir_labels:
                    st.error("Select at least one BEIR dataset in BEIR mix mode.")
                    return
                source_map = {s["label"]: s for s in beir_sources}
                selected_sources = [source_map[label] for label in selected_beir_labels if label in source_map]
                with st.spinner("Loading BEIR mix embeddings from indexed collections..."):
                    embeddings, metadatas, total_count = _load_beir_mix_embeddings(
                        selected_sources=selected_sources,
                        sample_limit=sample_limit,
                    )
            else:
                local_paths = [
                    local_label_to_path[label]
                    for label in selected_raw_local_labels
                    if label in local_label_to_path
                ]
                if not local_paths and not selected_raw_beir and not selected_raw_hf:
                    st.error("Select at least one source in Raw mixed sample mode.")
                    return

                progress_msg.info("Starting raw mixed sampling...")
                embeddings, metadatas, total_count = _load_raw_mix_embeddings(
                    local_paths=local_paths,
                    beir_datasets=selected_raw_beir,
                    hf_specs=selected_raw_hf,
                    beir_split=raw_beir_split,
                    per_source_sample_size=raw_per_source_sample_size,
                    hf_max_scan_examples=int(raw_hf_max_scan_examples),
                    batch_size=int(raw_embed_batch_size),
                    seed=int(raw_seed),
                    progress_callback=_raw_progress,
                )

            progress.progress(1.0)
            progress_msg.success("Run complete.")

            st.session_state["embedding_dashboard_last_run_config"] = current_run_config
            st.session_state["embedding_dashboard_embeddings"] = embeddings
            st.session_state["embedding_dashboard_metadatas"] = metadatas
            st.session_state["embedding_dashboard_total_count"] = total_count
        except Exception as exc:  # pragma: no cover - UI-only
            st.error(f"Failed to load embeddings: {exc}")
            return

    if not has_previous_run and not run_clicked:
        st.info("Set your sources and click **Run Analysis**.")
        return

    if has_previous_run and not run_clicked:
        last_run_config = st.session_state.get("embedding_dashboard_last_run_config", {})
        if current_run_config != last_run_config:
            st.warning("Settings changed. Click **Run Analysis** to refresh results.")

    embeddings = st.session_state["embedding_dashboard_embeddings"]
    metadatas = st.session_state["embedding_dashboard_metadatas"]
    total_count = st.session_state["embedding_dashboard_total_count"]

    stats = compute_basic_stats(embeddings)
    n_samples = stats["n_samples"]
    dim = stats["dim"]

    cols = st.columns(3)
    with cols[0]:
        st.metric("Total indexed vectors", value=str(total_count) if total_count is not None else "unknown")
    with cols[1]:
        st.metric("Sample size", value=str(sample_limit))
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
    source_labels: List[str] = []
    for meta in metadatas:
        if not isinstance(meta, dict):
            meta = {}
        path_str = str(meta.get("filepath") or "")
        filepaths.append(path_str)
        source_labels.append(str(meta.get("source_label") or "selected-source"))
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
            "source_label": source_labels,
        }
    )

    st.caption(
        "PCA is computed on the sampled embeddings only, using a "
        "pure NumPy SVD. Explained variance ratios for PC1/PC2: "
        f"{explained[0]:.4f}, {explained[1]:.4f}."
    )

    color_by = st.selectbox("Color points by", ["folder", "source_label", "none"], index=0)

    if color_by == "folder":
        st.scatter_chart(df_pca, x="pc1", y="pc2", color="folder")
    elif color_by == "source_label":
        st.scatter_chart(df_pca, x="pc1", y="pc2", color="source_label")
    else:
        st.scatter_chart(df_pca, x="pc1", y="pc2")

    with st.expander("Raw PCA sample (first 200 rows)"):
        st.dataframe(df_pca.head(200), use_container_width=True)

    st.subheader("PCA projection (lowest-variance dimensions)")
    if dim < 2:
        st.info("Need at least 2 embedding dimensions for low-variance PCA.")
    else:
        default_low_k = min(128, dim)
        min_low_k = 2 if dim <= 8 else 8
        low_k = st.slider(
            "Number of lowest-variance dimensions to keep",
            min_value=min_low_k,
            max_value=dim,
            value=max(min_low_k, default_low_k),
            step=1,
            key="low_var_pca_k",
        )

        low_var_order = np.argsort(var)[:low_k]
        low_var_embeddings = embeddings[:, low_var_order]

        try:
            low_projected, low_explained = compute_pca(low_var_embeddings, n_components=2)
        except Exception as exc:  # pragma: no cover - UI-only
            st.error(f"Failed to compute low-variance PCA: {exc}")
            return

        df_low_pca = pd.DataFrame(
            {
                "pc1": low_projected[:, 0],
                "pc2": low_projected[:, 1],
                "filepath": filepaths,
                "folder": folders,
                "source_label": source_labels,
            }
        )

        st.caption(
            "This PCA is computed after restricting each embedding to the "
            f"{low_k} lowest-variance dimensions. Explained variance ratios "
            f"for PC1/PC2: {low_explained[0]:.4f}, {low_explained[1]:.4f}."
        )

        if color_by == "folder":
            st.scatter_chart(df_low_pca, x="pc1", y="pc2", color="folder")
        elif color_by == "source_label":
            st.scatter_chart(df_low_pca, x="pc1", y="pc2", color="source_label")
        else:
            st.scatter_chart(df_low_pca, x="pc1", y="pc2")

        with st.expander("Raw low-variance PCA sample (first 200 rows)"):
            st.dataframe(df_low_pca.head(200), use_container_width=True)

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
