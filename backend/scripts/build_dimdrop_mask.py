from __future__ import annotations

"""Build a reusable dim-drop mask (dimension order) from a BEIR corpus.

Usage (from backend/):

    source ../.venv/bin/activate
    pip install .[benchmark]
    python scripts/build_dimdrop_mask.py --dataset scifact --reindex

This writes:
- dim-order file: SMARTFILES_DATA_DIR/benchmarks/beir/<dataset>/dimdrop_dim_order.npy
- metadata file: SMARTFILES_DATA_DIR/benchmarks/beir/<dataset>/dimdrop_dim_order.meta.json

Then set one of:
- SMARTFILES_DIMDROP_MASK_PATH=<absolute path to .npy>
- SMARTFILES_DIMDROP_BEIR_DATASET=<dataset>

The API will load the saved dim-order and use it for dim-drop scoring.
"""

import argparse
from pathlib import Path
from smartfiles.benchmarks.dimdrop_mask import build_beir_dimdrop_mask


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BEIR-based dim-drop mask")
    parser.add_argument("--dataset", required=True, help="BEIR dataset name (e.g., scifact)")
    parser.add_argument("--split", default="test", help="BEIR split (default: test)")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Max number of embeddings sampled to estimate per-dim variance",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used when reindexing BEIR corpus",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Download/reindex BEIR corpus before building the mask",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path for .npy dim-order file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    try:
        npy_path, _meta_path = build_beir_dimdrop_mask(
            dataset=args.dataset,
            split=args.split,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            reindex=args.reindex,
            output_path=output_path,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"[dimdrop-mask] set SMARTFILES_DIMDROP_MASK_PATH={npy_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
