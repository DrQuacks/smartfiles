from __future__ import annotations

from typing import Dict, Mapping, Sequence
from datetime import datetime, timezone
import time

from beir.retrieval.evaluation import EvaluateRetrieval

from .models import RunConfig, RunResult
from ..backends.base import RetrievalBackend
from ..datasets.beir import Corpus, Queries, Qrels


def _build_results_dict(
    backend: RetrievalBackend,
    queries: Queries,
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    """Run retrieval for all queries and build BEIR-style results dict."""

    results: Dict[str, Dict[str, float]] = {}
    hits_by_qid = backend.bulk_search(queries, top_k=top_k)
    for qid, hits in hits_by_qid.items():
        results[qid] = {hit.doc_id: float(hit.score) for hit in hits}
    return results


def evaluate_beir_run(
    backend: RetrievalBackend,
    corpus: Corpus,
    queries: Queries,
    qrels: Qrels,
    config: RunConfig,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RunResult:
    """Index corpus on the backend and evaluate using BEIR metrics."""
    start = time.perf_counter()

    backend.index_corpus(corpus)
    results = _build_results_dict(backend, queries, top_k=config.top_k)

    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels=qrels,
        results=results,
        k_values=list(k_values),
    )

    metrics = {
        "ndcg": {str(k): float(ndcg.get(f"NDCG@{k}", 0.0)) for k in k_values},
        "map": {str(k): float(_map.get(f"MAP@{k}", 0.0)) for k in k_values},
        "recall": {str(k): float(recall.get(f"Recall@{k}", 0.0)) for k in k_values},
        "precision": {str(k): float(precision.get(f"P@{k}", 0.0)) for k in k_values},
    }

    backend_metadata = {"backend_name": backend.name}
    # Allow backends to attach additional metadata (e.g. embedding
    # configuration) via an optional `metadata` dict attribute.
    extra_meta = getattr(backend, "metadata", None)
    if isinstance(extra_meta, dict):
        backend_metadata.update(extra_meta)

    ts = datetime.now(timezone.utc).isoformat()
    duration = time.perf_counter() - start

    return RunResult(
        config=config,
        timestamp=ts,
        duration_seconds=duration,
        metrics=metrics,
        backend_metadata=backend_metadata,
    )
