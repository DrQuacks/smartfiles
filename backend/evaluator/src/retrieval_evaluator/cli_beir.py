from __future__ import annotations

import argparse
from pathlib import Path

from .core.beir_evaluator import evaluate_beir_run
from .core.models import RunConfig
from .datasets.beir import BeirDataset
from .logging.jsonl_logger import JsonlRunLogger
from .backends.base import RetrievalBackend


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("retrieval-evaluator-beir")
    parser.add_argument("dataset", help="Name of the BEIR dataset (e.g. scifact)")
    parser.add_argument("data_dir", help="Path to the BEIR dataset directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("-k", "--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path.home() / ".retrieval_evaluator" / "beir_runs.jsonl",
    )
    return parser


def run_with_backend(backend: RetrievalBackend) -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = BeirDataset(name=args.dataset, data_dir=args.data_dir)
    corpus, queries, qrels = dataset.load(split=args.split)

    config = RunConfig(
        dataset=args.dataset,
        split=args.split,
        top_k=args.top_k,
        batch_size=args.batch_size,
        backend_name=backend.name,
        tag=args.tag,
    )

    result = evaluate_beir_run(
        backend=backend,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        config=config,
    )

    logger = JsonlRunLogger(args.log_path)
    logger.append([result])

    print(f"Dataset: {config.dataset} | Backend: {backend.name}")
    for metric_name, bucket in result.metrics.items():
        pretty = ", ".join(f"{metric_name}@{k}={v:.4f}" for k, v in sorted(bucket.items(), key=lambda kv: int(kv[0])))
        print(f"  {pretty}")
