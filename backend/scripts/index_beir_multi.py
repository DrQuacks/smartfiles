from __future__ import annotations

"""Index a curated set of diverse BEIR datasets for dim-drop mask building.

These datasets span a wide range of domains and styles:
- scifact      : scientific claim verification (biomedical)
- nfcorpus     : nutritional fact retrieval (short queries, long docs)
- fiqa          : financial QA from StackExchange
- quora         : duplicate question retrieval
- dbpedia-entity: entity retrieval from DBpedia (broad general knowledge)
- trec-covid    : COVID-19 biomedical retrieval

Run from backend/:

    source ../.venv/bin/activate
    pip install .[benchmark]
    python scripts/index_beir_multi.py

After this finishes, re-open the embedding dashboard. All indexed
datasets will appear in the BEIR mix dropdown automatically.
"""

import argparse

from smartfiles.benchmarks.beir_runner import index_beir_corpus

# Diverse set: different domains, query types, and document lengths.
# Deliberately NOT just science — mixes general knowledge, finance,
# web forums, and biomedical so the variance estimate reflects
# the full breadth of natural language, not one domain.
DATASETS = [
    ("scifact",        "test"),   # scientific fact checking
    ("nfcorpus",       "test"),   # nutrition / health
    ("fiqa",           "train"),  # financial QA
    ("quora",          "dev"),    # question deduplication
    ("dbpedia-entity", "test"),   # general entity knowledge
    ("trec-covid",     "test"),   # biomedical (COVID)
]

BATCH_SIZE = 128


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a curated multi-dataset BEIR set")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Embedding batch size used while indexing each dataset",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default="",
        help="Stop once this dataset finishes indexing (e.g. quora)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated list of dataset names to index",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate each dataset index before ingesting (destructive)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    batch_size = int(args.batch_size)
    stop_after = args.stop_after.strip().lower()
    recreate_index = bool(args.recreate)

    selected = DATASETS
    if args.only.strip():
        wanted = {name.strip().lower() for name in args.only.split(",") if name.strip()}
        selected = [(name, split) for name, split in DATASETS if name.lower() in wanted]
        if not selected:
            raise SystemExit(f"No matching datasets for --only={args.only!r}")

    for dataset, split in selected:
        print(f"\n{'=' * 60}")
        print(f"Indexing BEIR dataset: {dataset!r} (split={split})")
        print(f"{'=' * 60}")
        try:
            store, _embedder, corpus, _queries, _qrels = index_beir_corpus(
                dataset_name=dataset,
                split=split,
                batch_size=batch_size,
                recreate_index=recreate_index,
            )
            count = store._collection.count()
            print(f"  ✓ Indexed {count} documents into beir-{dataset}")
        except Exception as exc:
            print(f"  ✗ Failed to index {dataset}: {exc}")

        if stop_after and dataset.lower() == stop_after:
            print(f"\nStopping early after dataset={dataset!r} as requested.")
            break

    print("\nAll done. Re-open the embedding dashboard to see the new datasets.")


if __name__ == "__main__":
    main()
