from __future__ import annotations

"""Run a small matrix of BEIR benchmarks in a structured way.

Usage (from backend/ with benchmark extra installed):

    source ../.venv/bin/activate
    python scripts/run_beir_matrix.py

This script relies on the optional BEIR dependencies (`pip install .[benchmark]`).
"""

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

from smartfiles.benchmarks.beir_runner import run_beir_benchmark


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str
    split: str = "test"
    top_k: int = 10
    batch_size: int = 128
    skip_index: bool = False
    embedding_profile: str | None = None
    tag: str | None = None


def run_benchmarks(configs: Sequence[BenchmarkConfig]) -> None:
    for cfg in configs:
        env = os.environ.copy()
        if cfg.embedding_profile is not None:
            env["SMARTFILES_EMBEDDING_PROFILE"] = cfg.embedding_profile
        # Print a clear separator so logs are easy to scan.
        print("=" * 80)
        print(
            f"Dataset={cfg.dataset} split={cfg.split} "
            f"profile={cfg.embedding_profile or env.get('SMARTFILES_EMBEDDING_PROFILE')} "
            f"top_k={cfg.top_k} batch_size={cfg.batch_size} skip_index={cfg.skip_index}"
        )
        print("=" * 80)

        # Apply env var in-process before calling the runner.
        if cfg.embedding_profile is not None:
            os.environ["SMARTFILES_EMBEDDING_PROFILE"] = cfg.embedding_profile

        run_beir_benchmark(
            dataset_name=cfg.dataset,
            split=cfg.split,
            top_k=cfg.top_k,
            batch_size=cfg.batch_size,
            skip_index=cfg.skip_index,
            run_tag=cfg.tag,
        )


def default_matrix() -> list[BenchmarkConfig]:
    """Return a small, editable matrix of benchmark runs.

    You can tweak this list to compare different datasets, models,
    and basic hyperparameters (top_k, batch_size). Each run will
    also be recorded in the BEIR runs.jsonl log with its tag.
    """

    profiles: Iterable[str | None] = [
        "all-minilm-l6-v2",
        "bge-small-en-v1",
        "bge-base-en-v1",
    ]

    datasets: Iterable[str] = [
        "scifact",
        # Add more BEIR datasets here if desired, e.g. "nfcorpus", "trec-covid".
    ]

    matrix: list[BenchmarkConfig] = []
    for dataset in datasets:
        for profile in profiles:
            if profile is None:
                tag = f"{dataset},default-profile"
            else:
                tag = f"{dataset},{profile}"
            matrix.append(
                BenchmarkConfig(
                    dataset=dataset,
                    split="test",
                    top_k=10,
                    batch_size=128,
                    skip_index=False,
                    embedding_profile=profile,
                    tag=tag,
                )
            )
    return matrix


if __name__ == "__main__":
    configs = default_matrix()
    run_benchmarks(configs)
