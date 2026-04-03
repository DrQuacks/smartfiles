from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from smartfiles.config import get_data_dir
from smartfiles.embeddings.embedding_model import (
    EmbeddingModel,
    get_default_embedding_model,
    PROFILE_ENV_VAR,
    MODEL_ENV_VAR,
)
from smartfiles.database.vector_store import ChromaVectorStore
from smartfiles.ingestion.chunker import DocumentChunk
from smartfiles.search.search_engine import run_search


def _beir_root_dir() -> Path:
    """Root directory for BEIR benchmark data under SMARTFILES_DATA_DIR.

    This keeps benchmark data completely separate from regular user
    indexes and corpus artifacts.
    """

    return get_data_dir() / "benchmarks" / "beir"


def _dataset_dir(dataset_name: str) -> Path:
    return _beir_root_dir() / dataset_name


def _runs_log_path() -> Path:
    """Return the path to the JSONL runs log for BEIR benchmarks."""

    return _beir_root_dir() / "runs.jsonl"


def _download_and_load_beir(dataset_name: str, split: str) -> Tuple[Dict, Dict, Dict]:
    """Download (if needed) and load a BEIR dataset.

    Returns (corpus, queries, qrels) as provided by BEIR.
    """

    out_dir = _dataset_dir(dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, str(out_dir))

    loader = GenericDataLoader(data_folder=data_path)
    corpus, queries, qrels = loader.load(split=split)
    return corpus, queries, qrels


def _build_beir_store(dataset_name: str) -> ChromaVectorStore:
    db_dir = _dataset_dir(dataset_name) / "database"
    return ChromaVectorStore(db_path=db_dir, collection_name=f"beir-{dataset_name}")


def index_beir_corpus(
    *,
    dataset_name: str,
    split: str = "test",
    embedder: EmbeddingModel | None = None,
    batch_size: int = 128,
    recreate_index: bool = True,
) -> Tuple[ChromaVectorStore, EmbeddingModel, Dict, Dict, Dict]:
    """Index a BEIR corpus into a dedicated Chroma collection.

    - Uses the current SmartFiles embedding model.
    - Stores embeddings in a separate database folder so user indexes
      are unaffected.
    - Treats each BEIR document as a single chunk with id == doc_id.
    """

    if embedder is None:
        embedder = get_default_embedding_model()

    corpus, queries, qrels = _download_and_load_beir(dataset_name, split)

    store = _build_beir_store(dataset_name)
    if recreate_index:
        store.reset()

    # Prepare and index documents in batches to avoid high memory usage.
    doc_items = list(corpus.items())
    texts_batch = []
    chunks_batch = []

    def _flush_batch() -> None:
        if not chunks_batch:
            return
        embeddings = embedder.embed_texts(texts_batch)
        store.add_documents(chunks_batch, embeddings)
        texts_batch.clear()
        chunks_batch.clear()

    for doc_id, fields in doc_items:
        title = (fields.get("title") or "").strip()
        body = (fields.get("text") or "").strip()
        combined = (title + "\n" + body).strip() if title or body else ""
        if not combined:
            continue

        chunk = DocumentChunk(
            id=str(doc_id),
            filepath=str(doc_id),
            chunk_index=0,
            text=combined,
        )
        texts_batch.append(combined)
        chunks_batch.append(chunk)

        if len(texts_batch) >= batch_size:
            _flush_batch()

    _flush_batch()

    return store, embedder, corpus, queries, qrels


def run_beir_benchmark(
    *,
    dataset_name: str,
    split: str = "test",
    top_k: int = 10,
    batch_size: int = 128,
    skip_index: bool = False,
    run_tag: str | None = None,
) -> None:
    """Run a BEIR benchmark against the current SmartFiles stack.

    This will:
    - download the requested BEIR dataset (if needed),
    - index its corpus into a dedicated Chroma collection, and
    - evaluate retrieval using SmartFiles' own search pipeline.

    Results are printed to stdout and do not affect regular user
    indexes or workflow.
    """

    # Ensure benchmark data root exists.
    _beir_root_dir().mkdir(parents=True, exist_ok=True)

    if skip_index:
        # We still need corpus, queries, qrels for evaluation. Assume
        # the corresponding Chroma collection already exists.
        corpus, queries, qrels = _download_and_load_beir(dataset_name, split)
        embedder = get_default_embedding_model()
        store = _build_beir_store(dataset_name)
    else:
        store, embedder, corpus, queries, qrels = index_beir_corpus(
            dataset_name=dataset_name,
            split=split,
            embedder=None,
            batch_size=batch_size,
            recreate_index=True,
        )

    max_k = max(1, top_k)

    # Run SmartFiles search for each BEIR query and collect scores.
    results: Dict[str, Dict[str, float]] = {}
    for qid, query in queries.items():
        query_text = query if isinstance(query, str) else str(query)
        hits = run_search(query=query_text, k=max_k, embedder=embedder, store=store)
        doc_scores: Dict[str, float] = {}
        for hit in hits:
            doc_id = str(hit.get("filepath") or hit.get("id"))
            score = float(hit.get("score", 0.0) or 0.0)
            doc_scores[doc_id] = score
        results[qid] = doc_scores

    k_values = sorted({1, 3, 5, 10, max_k})

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values=k_values)

    print(f"BEIR benchmark for dataset='{dataset_name}', split='{split}'")
    print(f"Top-K evaluated: {k_values}")
    print("\nNDCG@K:")
    for k in k_values:
        val = ndcg.get(f"NDCG@{k}")
        if val is not None:
            print(f"  @{k}: {val:.4f}")

    print("\nRecall@K:")
    for k in k_values:
        val = recall.get(f"Recall@{k}")
        if val is not None:
            print(f"  @{k}: {val:.4f}")

    print("\nMAP@K:")
    for k in k_values:
        val = _map.get(f"MAP@{k}")
        if val is not None:
            print(f"  @{k}: {val:.4f}")

    print("\nPrecision@K:")
    for k in k_values:
        val = precision.get(f"P@{k}")
        if val is not None:
            print(f"  @{k}: {val:.4f}")

    print("\nDone.")

    # Persist a compact JSONL log entry with key metadata so runs
    # can be compared over time.
    try:
        import json
        import importlib.metadata

        smartfiles_version = None
        try:
            smartfiles_version = importlib.metadata.version("smartfiles")
        except importlib.metadata.PackageNotFoundError:
            smartfiles_version = None

        embedding_profile = os.getenv(PROFILE_ENV_VAR)
        embedding_model_override = os.getenv(MODEL_ENV_VAR)
        # Best-effort attempt to capture the underlying model id/path.
        model_name_or_path = getattr(getattr(embedder, "model", None), "name_or_path", None)

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_name,
            "split": split,
            "top_k": top_k,
            "batch_size": batch_size,
            "skip_index": skip_index,
            "k_values": k_values,
            "embedding": {
                "env_profile": embedding_profile,
                "env_model": embedding_model_override,
                "model_name_or_path": model_name_or_path,
            },
            "smartfiles_version": smartfiles_version,
            "run_tag": run_tag,
            "metrics": {
                "ndcg": {
                    str(k): ndcg.get(f"NDCG@{k}")
                    for k in k_values
                    if ndcg.get(f"NDCG@{k}") is not None
                },
                "map": {
                    str(k): _map.get(f"MAP@{k}")
                    for k in k_values
                    if _map.get(f"MAP@{k}") is not None
                },
                "recall": {
                    str(k): recall.get(f"Recall@{k}")
                    for k in k_values
                    if recall.get(f"Recall@{k}") is not None
                },
                "precision": {
                    str(k): precision.get(f"P@{k}")
                    for k in k_values
                    if precision.get(f"P@{k}") is not None
                },
            },
        }

        runs_path = _runs_log_path()
        runs_path.parent.mkdir(parents=True, exist_ok=True)
        with runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        # Logging should never cause the benchmark itself to fail.
        pass
