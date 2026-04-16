from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from smartfiles.benchmarks.beir_runner import index_beir_corpus
from smartfiles.config import get_data_dir
from smartfiles.database.vector_store import ChromaVectorStore
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
