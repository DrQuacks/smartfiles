from __future__ import annotations

import json
import importlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from smartfiles.benchmarks.beir_runner import _download_and_load_beir, index_beir_corpus
from smartfiles.config import get_data_dir
from smartfiles.database.text_store import iter_corpus_documents
from smartfiles.embeddings.embedding_model import EmbeddingModel, get_default_embedding_model
from smartfiles.database.vector_store import ChromaVectorStore
from smartfiles.folder_registry import list_folders
from smartfiles.search.dimdrop import compute_global_dim_order


def default_beir_mask_path(dataset: str) -> Path:
    return get_data_dir() / "benchmarks" / "beir" / dataset / "dimdrop_dim_order.npy"


def _beir_store(dataset: str) -> ChromaVectorStore:
    db_dir = get_data_dir() / "benchmarks" / "beir" / dataset / "database"
    return ChromaVectorStore(db_path=db_dir, collection_name=f"beir-{dataset}")


def build_beir_dimdrop_mask(
    *,
    dataset: str,
    split: str = "test",
    sample_size: int = 2000,
    batch_size: int = 128,
    reindex: bool = False,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    """Build and persist a BEIR-based dim-order mask artifact.

    Returns ``(npy_path, meta_json_path)``.
    """

    npy_path = output_path.expanduser().resolve() if output_path else default_beir_mask_path(dataset)
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    if reindex:
        print(f"[dimdrop-mask] indexing BEIR dataset='{dataset}' split='{split}'", flush=True)
        index_beir_corpus(
            dataset_name=dataset,
            split=split,
            batch_size=batch_size,
            recreate_index=True,
        )

    store = _beir_store(dataset)
    print(
        f"[dimdrop-mask] computing dim order from BEIR store (dataset={dataset}, sample_size={sample_size})",
        flush=True,
    )
    dim_order = compute_global_dim_order(store, max_sample=sample_size)
    if dim_order is None:
        raise ValueError(
            "Could not compute dim-order: BEIR index appears empty. "
            "Run with reindex=True or verify benchmark data exists."
        )

    np.save(npy_path, dim_order)

    meta = {
        "dataset": dataset,
        "split": split,
        "sample_size": int(sample_size),
        "dim_count": int(np.asarray(dim_order).shape[0]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "beir",
        "output_path": str(npy_path),
    }
    meta_path = npy_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[dimdrop-mask] wrote {npy_path}", flush=True)
    print(f"[dimdrop-mask] wrote {meta_path}", flush=True)

    return npy_path, meta_path


def default_sampled_mask_path(label: str) -> Path:
    safe = label.replace(",", "-").replace("/", "-").replace(" ", "-")
    return get_data_dir() / "benchmarks" / "beir" / "masks" / safe / "dimdrop_dim_order.npy"


def _sample_corpus_texts(
    *,
    dataset: str,
    split: str,
    max_texts: int,
    seed: int,
) -> list[str]:
    corpus, _queries, _qrels = _download_and_load_beir(dataset, split)

    texts: list[str] = []
    for fields in corpus.values():
        title = str((fields.get("title") or "")).strip()
        body = str((fields.get("text") or "")).strip()
        combined = (title + "\n" + body).strip() if title or body else ""
        if combined:
            texts.append(combined)

    if len(texts) <= max_texts:
        return texts

    rng = random.Random(seed)
    indices = rng.sample(range(len(texts)), k=max_texts)
    return [texts[i] for i in indices]


def _sample_local_corpus_texts(
    *,
    root_folder: Path,
    max_texts: int,
    seed: int,
) -> list[str]:
    texts: list[str] = []
    for _original_path, text in iter_corpus_documents(root_folder):
        normalized = text.strip()
        if normalized:
            texts.append(normalized)

    if len(texts) <= max_texts:
        return texts

    rng = random.Random(seed)
    indices = rng.sample(range(len(texts)), k=max_texts)
    return [texts[i] for i in indices]


def _parse_hf_spec(spec: str) -> tuple[str, str | None, str, str]:
    parts = [part.strip() for part in spec.split("::")]
    repo_id = parts[0] if parts else ""
    if not repo_id:
        raise ValueError(f"Invalid Hugging Face dataset spec: {spec!r}")

    config = parts[1] or None if len(parts) > 1 else None
    split = parts[2] or "train" if len(parts) > 2 else "train"
    text_field = parts[3] or "text" if len(parts) > 3 else "text"
    return repo_id, config, split, text_field


def _sample_hf_dataset_texts(
    *,
    spec: str,
    max_texts: int,
    seed: int,
    max_scan_examples: int = 20000,
) -> list[str]:
    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Hugging Face dataset sampling requires the 'datasets' package. "
            "Install the benchmark extra with `pip install .[benchmark]`."
        ) from exc

    repo_id, config, split, text_field = _parse_hf_spec(spec)
    dataset = load_dataset(repo_id, name=config, split=split, streaming=True)

    rng = random.Random(seed)
    reservoir: list[str] = []
    seen_valid = 0

    for index, example in enumerate(dataset):
        if index >= max_scan_examples:
            break

        value = example.get(text_field)
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


def _dedupe_texts(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for text in texts:
        normalized = text.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _compute_dim_order_from_embeddings(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("Need at least two embeddings to compute dim order")
    std = matrix.std(axis=0)
    return np.argsort(std ** 2)


def build_beir_sampled_dimdrop_mask(
    *,
    datasets: Iterable[str],
    split: str = "test",
    per_dataset_sample_size: int = 500,
    batch_size: int = 128,
    seed: int = 13,
    embedder: EmbeddingModel | None = None,
    output_path: Path | None = None,
    label: str | None = None,
) -> tuple[Path, Path]:
    """Build a dim-drop mask from sampled BEIR raw texts without indexing.

    This is the preferred path for experimentation because it avoids
    indexing entire benchmark corpora into Chroma. Instead, it:
    1) downloads/loads each BEIR dataset,
    2) samples a capped number of raw documents per dataset,
    3) embeds those sampled texts directly,
    4) computes per-dimension variance over the combined sample.
    """

    dataset_list = [d.strip() for d in datasets if d.strip()]
    if not dataset_list:
        raise ValueError("At least one dataset is required")

    if embedder is None:
        embedder = get_default_embedding_model()

    label_value = label or ",".join(dataset_list)
    npy_path = output_path.expanduser().resolve() if output_path else default_sampled_mask_path(label_value)
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    all_texts: list[str] = []
    sample_counts: dict[str, int] = {}
    for offset, dataset in enumerate(dataset_list):
        texts = _sample_corpus_texts(
            dataset=dataset,
            split=split,
            max_texts=per_dataset_sample_size,
            seed=seed + offset,
        )
        sample_counts[dataset] = len(texts)
        all_texts.extend(texts)
        print(
            f"[dimdrop-mask] sampled {len(texts)} texts from dataset='{dataset}' split='{split}'",
            flush=True,
        )

    if len(all_texts) < 2:
        raise ValueError("Need at least two sampled texts across datasets")

    vectors: list[list[float]] = []
    for start in range(0, len(all_texts), batch_size):
        batch = all_texts[start : start + batch_size]
        print(
            f"[dimdrop-mask] embedding batch {start}-{start + len(batch) - 1} / {len(all_texts) - 1}",
            flush=True,
        )
        vectors.extend(embedder.embed_texts(batch))

    matrix = np.asarray(vectors, dtype=np.float32)
    dim_order = _compute_dim_order_from_embeddings(matrix)
    np.save(npy_path, dim_order)

    meta = {
        "datasets": dataset_list,
        "split": split,
        "per_dataset_sample_size": int(per_dataset_sample_size),
        "total_sample_count": int(matrix.shape[0]),
        "dim_count": int(matrix.shape[1]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "beir-sampled",
        "sample_counts": sample_counts,
        "output_path": str(npy_path),
        "seed": int(seed),
    }
    meta_path = npy_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[dimdrop-mask] wrote {npy_path}", flush=True)
    print(f"[dimdrop-mask] wrote {meta_path}", flush=True)
    return npy_path, meta_path


def build_mixed_sampled_dimdrop_mask(
    *,
    beir_datasets: Iterable[str] = (),
    hf_datasets: Iterable[str] = (),
    local_folders: Iterable[Path | str] = (),
    include_registered_local: bool = False,
    beir_split: str = "test",
    per_source_sample_size: int = 500,
    hf_max_scan_examples: int = 20000,
    batch_size: int = 128,
    seed: int = 13,
    embedder: EmbeddingModel | None = None,
    output_path: Path | None = None,
    label: str | None = None,
) -> tuple[Path, Path]:
    """Build a dim-drop mask from a mixed, non-indexed text sample.

    Sources can include:
    - sampled BEIR raw corpora,
    - sampled local SmartFiles raw-text corpora,
    - sampled streaming Hugging Face datasets.

    This is the preferred recovery path when you want a broader mask
    corpus without indexing large external collections into Chroma.
    """

    if embedder is None:
        embedder = get_default_embedding_model()

    source_entries: list[tuple[str, str, int]] = []

    for dataset in beir_datasets:
        cleaned = dataset.strip()
        if cleaned:
            source_entries.append(("beir", cleaned, seed + len(source_entries)))

    for spec in hf_datasets:
        cleaned = spec.strip()
        if cleaned:
            source_entries.append(("hf", cleaned, seed + len(source_entries)))

    seen_local_paths: set[str] = set()
    for folder in local_folders:
        path = Path(folder).expanduser().resolve()
        normalized = str(path)
        if normalized not in seen_local_paths:
            seen_local_paths.add(normalized)
            source_entries.append(("local", normalized, seed + len(source_entries)))

    if include_registered_local:
        for entry in list_folders():
            normalized = str(Path(entry.path).expanduser().resolve())
            if normalized not in seen_local_paths:
                seen_local_paths.add(normalized)
                source_entries.append(("local", normalized, seed + len(source_entries)))

    if not source_entries:
        raise ValueError("At least one source is required")

    label_value = label or "mixed-sampled"
    npy_path = output_path.expanduser().resolve() if output_path else default_sampled_mask_path(label_value)
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    all_texts: list[str] = []
    sample_counts: dict[str, int] = {}
    source_kinds: dict[str, str] = {}

    for kind, value, source_seed in source_entries:
        if kind == "beir":
            texts = _sample_corpus_texts(
                dataset=value,
                split=beir_split,
                max_texts=per_source_sample_size,
                seed=source_seed,
            )
            source_name = f"beir:{value}"
        elif kind == "hf":
            texts = _sample_hf_dataset_texts(
                spec=value,
                max_texts=per_source_sample_size,
                seed=source_seed,
                max_scan_examples=hf_max_scan_examples,
            )
            source_name = f"hf:{value}"
        elif kind == "local":
            texts = _sample_local_corpus_texts(
                root_folder=Path(value),
                max_texts=per_source_sample_size,
                seed=source_seed,
            )
            source_name = f"local:{value}"
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported source kind: {kind}")

        sample_counts[source_name] = len(texts)
        source_kinds[source_name] = kind
        all_texts.extend(texts)
        print(
            f"[dimdrop-mask] sampled {len(texts)} texts from {source_name}",
            flush=True,
        )

    unique_texts = _dedupe_texts(all_texts)
    if len(unique_texts) < 2:
        raise ValueError("Need at least two sampled texts across sources")

    vectors: list[list[float]] = []
    for start in range(0, len(unique_texts), batch_size):
        batch = unique_texts[start : start + batch_size]
        print(
            f"[dimdrop-mask] embedding batch {start}-{start + len(batch) - 1} / {len(unique_texts) - 1}",
            flush=True,
        )
        vectors.extend(embedder.embed_texts(batch))

    matrix = np.asarray(vectors, dtype=np.float32)
    dim_order = _compute_dim_order_from_embeddings(matrix)
    np.save(npy_path, dim_order)

    meta = {
        "beir_datasets": [d.strip() for d in beir_datasets if d.strip()],
        "hf_datasets": [d.strip() for d in hf_datasets if d.strip()],
        "local_folders": sorted(seen_local_paths),
        "include_registered_local": bool(include_registered_local),
        "beir_split": beir_split,
        "per_source_sample_size": int(per_source_sample_size),
        "hf_max_scan_examples": int(hf_max_scan_examples),
        "raw_total_sample_count": int(len(all_texts)),
        "deduped_total_sample_count": int(matrix.shape[0]),
        "dim_count": int(matrix.shape[1]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "mixed-sampled",
        "sample_counts": sample_counts,
        "source_kinds": source_kinds,
        "output_path": str(npy_path),
        "seed": int(seed),
    }
    meta_path = npy_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[dimdrop-mask] wrote {npy_path}", flush=True)
    print(f"[dimdrop-mask] wrote {meta_path}", flush=True)
    return npy_path, meta_path
