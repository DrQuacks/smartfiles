# SmartFiles Benchmarking (BEIR)

This backend includes an **optional** benchmarking harness based on the
[BEIR](https://github.com/beir-cellar/beir) evaluation suite. It lets you
measure how the current SmartFiles embedding model + Chroma setup performs
on standard retrieval benchmarks, without touching your regular indexes
or day-to-day SmartFiles workflow.

Benchmark data and indexes live under a separate directory:

- `~/.smartfiles/benchmarks/beir/<dataset>/`

so they do not interfere with your normal `~/.smartfiles/database`.

## Installation

The BEIR harness is optional. To use it, install the `benchmark` extra
from the backend directory:

```bash
cd backend
source .venv/bin/activate
pip install .[benchmark]
```

This pulls in the `beir` library and its dependencies, leaving the core
SmartFiles install unchanged.

## Running a benchmark

After installing the extra, you can run a BEIR benchmark via the CLI:

```bash
cd backend
source .venv/bin/activate
smartfiles benchmark-beir scifact
```

The command will:

1. Download the specified BEIR dataset (if not already present).
2. Index its corpus into a dedicated Chroma collection under
   `~/.smartfiles/benchmarks/beir/<dataset>/database`.
3. Run SmartFiles search for all queries in the chosen split (default: `test`).
4. Report standard retrieval metrics (NDCG, Recall, MAP, Precision at K).

### Options

```bash
smartfiles benchmark-beir --help
```

Key options:

- `dataset` (argument): BEIR dataset name, e.g. `scifact`, `nfcorpus`, `trec-covid`.
- `--split`: Data split to use (default: `test`).
- `-k`, `--top-k`: Maximum cutoff K to evaluate (default: 10).
- `--batch-size`: Batch size when embedding the corpus (default: 128).
- `--skip-index`: Reuse an existing benchmark index instead of rebuilding.

You can also attach a free-form label to a run:

- `--tag`: Optional string label, e.g. `--tag "bge-small-en-v1,chunk500"`.

## Run logging and metadata

Each benchmark run appends a JSON line to:

- `~/.smartfiles/benchmarks/beir/runs.jsonl`

Every entry records key metadata so you can compare runs over time, including:

- Timestamp (UTC)
- Dataset and split
- `top_k`, `batch_size`, `skip_index`
- Embedding configuration:
   - `SMARTFILES_EMBEDDING_PROFILE` (if set)
   - `SMARTFILES_EMBEDDING_MODEL` (if set)
   - The underlying SentenceTransformers `name_or_path` (when available)
- Installed `smartfiles` package version (if discoverable)
- The optional `--tag` label
- Metrics: NDCG, Recall, MAP, Precision at the evaluated K values

## How it stays isolated

- Uses a separate data root under `SMARTFILES_DATA_DIR/benchmarks/beir`.
- Stores all BEIR embeddings in a dedicated Chroma collection (one per dataset).
- Never touches the main SmartFiles corpus or index.

This lets you experiment with models and pipeline changes on BEIR without
risking your normal SmartFiles usage.
