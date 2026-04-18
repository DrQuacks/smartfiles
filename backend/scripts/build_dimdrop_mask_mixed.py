from __future__ import annotations

"""Build a mixed dim-drop mask from local, BEIR, and HF raw texts.

Recommended usage (from `backend/`):

    source ../.venv/bin/activate
    pip install .[benchmark]
    python scripts/build_dimdrop_mask_mixed.py \
        --include-registered-local \
        --beir fiqa \
        --beir nfcorpus \
        --hf fancyzhx/ag_news \
        --hf google/wiki40b::en::train::text \
        --per-source-sample-size 500 \
        --label bootstrap-mix

This avoids Chroma indexing entirely. It samples a capped number of raw
texts per source, embeds them directly, deduplicates exact text matches,
and computes a variance-based dimension order.
"""

import argparse
from pathlib import Path

from smartfiles.benchmarks.dimdrop_mask import build_mixed_sampled_dimdrop_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed sampled dimdrop mask")
    parser.add_argument("--beir", action="append", default=[], help="Repeatable BEIR dataset name")
    parser.add_argument(
        "--hf",
        action="append",
        default=[],
        help="Repeatable HF dataset spec repo_id::config::split::text_field",
    )
    parser.add_argument(
        "--local",
        action="append",
        default=[],
        help="Repeatable local root folder whose extracted corpus should be sampled",
    )
    parser.add_argument("--include-registered-local", action="store_true")
    parser.add_argument("--beir-split", default="test")
    parser.add_argument("--per-source-sample-size", type=int, default=500)
    parser.add_argument("--hf-max-scan-examples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    npy_path, _meta_path = build_mixed_sampled_dimdrop_mask(
        beir_datasets=args.beir,
        hf_datasets=args.hf,
        local_folders=args.local,
        include_registered_local=args.include_registered_local,
        beir_split=args.beir_split,
        per_source_sample_size=args.per_source_sample_size,
        hf_max_scan_examples=args.hf_max_scan_examples,
        batch_size=args.batch_size,
        seed=args.seed,
        output_path=output_path,
        label=args.label.strip() or None,
    )

    print(f"Built mixed sampled dim-drop mask: {npy_path}")
    print(f"Set SMARTFILES_DIMDROP_MASK_PATH={npy_path} to use it in API")


if __name__ == "__main__":
    main()
