# SmartFiles Backend Commands

This document collects common commands for working on the backend.

## Environment

```bash
cd /Users/Kellar/Develop/Projects/smartfiles/backend
# Activate virtualenv (if not already active)
source .venv/bin/activate
```

Install the backend package in editable mode (Python dependencies):

```bash
pip install -e .
```

For image OCR support (PNG/JPG), install the Tesseract binary
(optional but recommended):

```bash
brew install tesseract
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

## Inspecting Raw Extracted Text

When you run `smartfiles extract` or the full `smartfiles index`
pipeline, the full per-document text (after PDF parsing / OCR, before
chunking) is written to a corpus directory under the SmartFiles data
dir, namespaced by the root folder's name.

By default, the data dir is:

```bash
~/.smartfiles
```

You can override it with the `SMARTFILES_DATA_DIR` environment
variable. Within the data dir, each root folder gets its own
`<folder_name>_rawText` tree:

- Corpus: `$SMARTFILES_DATA_DIR/<folder_name>_rawText/corpus/`
- Stats: `$SMARTFILES_DATA_DIR/<folder_name>_rawText/stats/`
- Structure: mirrors the folder you passed to the CLI; each document like
	`some/file.pdf` becomes `some/file.pdf.txt`.

On macOS you can open the default corpus location in Finder with (for a
folder named `Documents`, for example):

```bash
open ~/.smartfiles/Documents_rawText/corpus
```

## Backend Server (planned)

Once the FastAPI server is implemented in `smartfiles.server.api`, run:

```bash
uvicorn smartfiles.server.api:app --reload
```

(See `docs/context_for_ai.md` and `docs/search_system.md` for server design details.)

## Troubleshooting

- **NumPy ABI error (1.x vs 2.x):** if you see an error about a module
	compiled with NumPy 1.x not working with NumPy 2, pin NumPy inside the
	backend virtualenv:

	```bash
	cd /Users/Kellar/Develop/Projects/smartfiles/backend
	source .venv/bin/activate
	pip install "numpy<2"
	```

- **Folders with spaces in the path:** when passing a folder to the
	CLI on macOS, quote or escape spaces so Typer sees the correct
	directory, for example:

	```bash
	smartfiles extract "/Users/Kellar/Desktop/Tutoring Docs/Science"
	# or
	smartfiles extract /Users/Kellar/Desktop/Tutoring\ Docs/Science
	```
