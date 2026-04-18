from __future__ import annotations

"""Build a dim-drop mask from sampled BEIR raw texts, without indexing.

Recommended usage (from `backend/`):

    source ../.venv/bin/activate
    pip install .[benchmark]
    python scripts/build_dimdrop_mask_sampled.py \
        --datasets scifact,nfcorpus,fiqa,trec-covid \
        --per-dataset-sample-size 500 \
        --label stable-small

This downloads/loads each dataset, samples a capped number of raw
texts per dataset, embeds them directly, and computes a variance-based
dimension order. No Chroma indexing is required.
"""

import argparse
from pathlib import Path

from smartfiles.benchmarks.dimdrop_mask import build_beir_sampled_dimdrop_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sampled BEIR dimdrop mask")
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated BEIR datasets, e.g. scifact,nfcorpus,fiqa,trec-covid",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-dataset-sample-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    npy_path, _meta_path = build_beir_sampled_dimdrop_mask(
        datasets=datasets,
        split=args.split,
        per_dataset_sample_size=args.per_dataset_sample_size,
        batch_size=args.batch_size,
        seed=args.seed,
        output_path=output_path,
        label=args.label.strip() or None,
    )

    print(f"Built sampled dim-drop mask: {npy_path}")
    print(f"Set SMARTFILES_DIMDROP_MASK_PATH={npy_path} to use it in API")


if __name__ == "__main__":
    main()
