# SmartFiles Backend Overview

This document summarizes the backend pipeline and the main modules
involved in ingestion, storage, embeddings, and search.

## High-level pipeline

```text
            +-----------------------------+
            |  Source folder (root)       |
            |  PDFs / images / DOCX       |
            +--------------+--------------+
                           |
                           v
                 [smartfiles extract]
                           |
                           v
+--------------------------+----------------------------+
| ingestion/file_scanner.py + ingestion/text_extractor.py |
|  - Find supported files   |  - Parse PDF/image/DOCX     |
|  - .pdf/.png/.jpg/.docx   |  - PDF text layer + OCR     |
+--------------------------+----------------------------+
                           |
                           v
                database/text_store.py
            (per-root `<name>_rawText/` tree)

  corpus/   stats/                     chunks/
  -------   ------                     -------
  - full    - extraction_*.txt         - optional per-chunk
    text      stats                      text files

                           |
                           v
                 [smartfiles chunk-from-text]
                           |
                           v
                ingestion/chunker.py
  - Reads corpus text (with PDF page markers)
  - Produces `DocumentChunk`s with page_start/page_end
  - Optional: writes chunk `.txt` files via text_store

                           |
                           v
           [smartfiles index-from-text / index]
                           |
                           v
     embeddings/embedding_model.py + database/vector_store.py
  - Load SentenceTransformer model (configurable profiles)
  - Embed chunks and store embeddings + metadata in Chroma

                           |
                           v
                 [smartfiles search]
                           |
                           v
                search/search_engine.py
  - Embed query
  - Query Chroma
  - Return ranked chunks with filepath + page info
```

## Modules and responsibilities

### `smartfiles/ingestion/`

- `file_scanner.py`
  - Recursively finds supported files under a root folder
    (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.docx`).
  - Used by `extract_documents` to know what to parse.

- `text_extractor.py`
  - `DefaultTextExtractor` handles:
    - PDFs via `pypdf` text layer first, then OCR fallback
      (`pdf2image` + Tesseract) when needed.
    - Images via Pillow + Tesseract (with a stronger math-aware pass
      when the initial OCR looks garbled).
    - DOCX via `python-docx` (paragraphs joined with newlines).
  - Inserts lightweight page markers into PDF text
    (e.g. `[[[SMARTFILES_PAGE 3]]]`) so later stages can recover page
    ranges for chunks.

- `chunker.py`
  - `DocumentChunk` dataclass: `id`, `filepath`, `chunk_index`, `text`,
    and optional `page_start`/`page_end`.
  - Parses corpus text, recognizes PDF page markers, and emits chunks
    with word-based chunking (default 500 words, 50-word overlap).

- `indexer.py`
  - `extract_documents(...)`:
    - Ensures a per-root registry entry.
    - Optionally recreates the corpus.
    - Iterates supported files and uses `DefaultTextExtractor`.
    - Writes full-text `.txt` files to the corpus and an
      `extraction_XXXX.txt` stats file.
    - Skips re-extraction for files that already have corpus text
      when `recreate_text=False`.
  - `chunk_corpus_from_text(...)`:
    - Reads corpus and runs `chunk_document`, optionally writing
      chunk `.txt` files, but **no embeddings**.
  - `build_index_from_corpus(...)`:
    - Reads corpus, chunks documents, embeds chunks, and writes
      into the vector store.
  - `run_indexing_pipeline(...)`:
    - Orchestrates the full pipeline: extract → chunk → embed/index.

### `smartfiles/database/`

- `text_store.py`
  - Knows the on-disk layout under `SMARTFILES_DATA_DIR`:
    - `<DATA_DIR>/<folder_name>_rawText/corpus/`
    - `<DATA_DIR>/<folder_name>_rawText/stats/`
    - `<DATA_DIR>/<folder_name>_rawText/chunks/`
  - Saves and iterates corpus documents and chunk files.
  - Generates incremental extraction stats filenames.

- `vector_store.py`
  - Wraps Chroma as a persistent vector database under
    `<DATA_DIR>/database`.
  - Stores chunk texts, embeddings, and metadata (filepath,
    chunk index, page range).
  - Exposes `search(...)` that returns normalized similarity
    scores in `[0, 100]`.

### `smartfiles/embeddings/`

- `embedding_model.py`
  - Central abstraction for embeddings.
  - `SupportedEmbeddingModel` registry with profiles like:
    - `all-minilm-l6-v2` (default): `sentence-transformers/all-MiniLM-L6-v2`.
    - `bge-small-en-v1`, `bge-base-en-v1` (optional BGE models).
  - Env configuration:
    - `SMARTFILES_EMBEDDING_MODEL`: explicit HF id or local path.
    - `SMARTFILES_EMBEDDING_PROFILE`: profile key from the registry.
  - `get_default_embedding_model()` returns an `EmbeddingModel` that
    exposes a single method `embed_texts(texts)` used everywhere else.

### `smartfiles/search/`

- `search_engine.py`
  - `run_search(query, k=5)`:
    - Loads the embedding model and vector store.
    - Embeds the query, performs a Chroma search, and returns
      result dicts including `text`, `score`, and metadata.
    - Optional profiling (if `SMARTFILES_PROFILE_SEARCH` is set)
      to show where time is spent: model load, store init,
      embedding, and vector search.

This overview is meant as a quick map of how the backend pieces fit
so you (or future collaborators) can jump into any part of the
pipeline without having to re-discover the structure from scratch.
