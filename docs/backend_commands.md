# SmartFiles Backend Commands

This document collects common commands for working on the backend.

## Environment

```bash
cd /Users/Kellar/Develop/Projects/smartfiles/backend
# Activate virtualenv (if not already active)
source .venv/bin/activate
```

Install the backend package in editable mode (and dependencies):

```bash
pip install -e .
```

## CLI (Indexing & Search)

Run full pipeline (extract → chunk → embed → index):

```bash
smartfiles index <folder>            # end-to-end
smartfiles index --recreate <folder> # rebuild corpus + index from scratch
```

Extract-only (debug PDF parsing / OCR):

```bash
smartfiles extract <folder>
smartfiles extract --recreate-text <folder> # wipe and rebuild corpus
```

Index-from-text only (chunk + embed + store):

```bash
smartfiles index-from-text <folder>
smartfiles index-from-text --recreate <folder> # rebuild index only
```

Search existing index:

```bash
smartfiles search "your query here"        # default top-k = 5
smartfiles search -k 10 "your query here" # custom k
```

## Backend Server (planned)

Once the FastAPI server is implemented in `smartfiles.server.api`, run:

```bash
uvicorn smartfiles.server.api:app --reload
```

(See `docs/context_for_ai.md` and `docs/search_system.md` for server design details.)
