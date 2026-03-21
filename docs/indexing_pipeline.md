# SmartFiles Indexing Pipeline

The indexing pipeline converts documents into searchable vector embeddings.

## Steps

### 1. File Scanning
Detect new or modified files.

Initial supported types (implemented):
- PDF
- PNG
- JPG
- DOCX

### 2. Text Extraction
PDF:
- Text layer parsing using pypdf.
- If the text layer is empty, pages are rendered to images and passed through OCR (standard, then stronger fallback when needed).

PNG / JPG:
- OCR using pytesseract (via Pillow image loading).

DOCX:
- Parsed using python-docx to extract paragraph text.

If OCR dependencies are not installed, image/PDF OCR is skipped (no text extracted for those files).

### 3. Text Cleaning
Normalize text:
- remove repeated headers
- normalize whitespace

### 4. Chunking

Current implementation uses simple word-based chunks as an approximation of tokens:

- chunk_size ≈ 500 words
- overlap ≈ 50 words

Small documents may use page-level chunks.

### 5. Metadata Extraction

Possible tags:
school
course
topic

### 6. Embedding

Example model:
bge-small-en

### 7. Storage

Embeddings and metadata stored in vector database.

In addition, the full raw text for each document is written to a
per-folder "corpus" directory for inspection and debugging:

- Base path: `<DATA_DIR>/<folder_name>_rawText/corpus/`, where
	`DATA_DIR` comes from the `SMARTFILES_DATA_DIR` environment variable
	(or defaults to `~/.smartfiles` when unset).
- Structure: mirrors the folder passed to the CLI, with files saved as
	UTF-8 `.txt` (filenames keep their original extension and add
	`.txt`, e.g. `file.pdf` → `file.pdf.txt`).

A parallel `stats/` directory
(`/<DATA_DIR>/<folder_name>_rawText/stats/`) contains per-run text
extraction summaries, including timestamp, git commit, and one entry
per file processed.

Optionally, per-chunk text files can also be written for inspection:

- Chunks path: `<DATA_DIR>/<folder_name>_rawText/chunks/`
- Structure: mirrors the folder you indexed; each document like
	`some/file.pdf` produces files such as
	`some/file.pdf.chunk-0.txt`, `some/file.pdf.chunk-1.txt`, etc.
- Control: enabled by default when running `smartfiles index` or
	`smartfiles index-from-text`, and can be disabled with
	`--no-save-chunks` if you don't want chunk files on disk.

This makes it easy to open a `.txt` file and validate that text
extraction (PDF parsing or OCR) is working as expected.

### 8. Pipeline Modes

The CLI exposes three ways to run the pipeline:

- `smartfiles extract <folder>` – only parse documents and write raw
	text files to the corpus.
- `smartfiles index-from-text <folder>` – read the existing corpus for
	`<folder>`, chunk, embed, and update the vector index.
- `smartfiles index <folder>` – run extraction and indexing together in
	one pass (recreates corpus and index when `--recreate` is used).
