# SmartFiles Backend Testing

This document tracks the tests for the backend and how they are organized.

## Test Types

- **Unit tests**: Exercise individual modules or small functions in isolation, using mocks where needed. Fast and deterministic.
- **Integration tests**: Exercise multiple components together (e.g., extraction pipeline + corpus + stats) using real files and a temporary `SMARTFILES_DATA_DIR`.
- **End-to-end tests**: Run the full CLI or future API stack against a small synthetic dataset, from `extract`/`index` through `search`.

## Layout

Backend tests live under `backend/tests`:

- `backend/tests/unit/` – unit tests
- `backend/tests/integration/` – integration tests
- `backend/tests/e2e/` – end-to-end tests (planned)

## Current Unit Tests

### `tests/unit/test_file_scanner.py`

- **Module under test**: `smartfiles.ingestion.file_scanner`
- **Purpose**: Verify that `list_files` finds only supported file types.
- **Key checks**:
  - Given a directory with a mix of files, only `.pdf`, `.png`, `.jpg`, `.jpeg`, and `.docx` are returned.

### `tests/unit/test_text_extractor.py`

- **Module under test**: `smartfiles.ingestion.text_extractor`
- **DOCX extraction**:
  - Builds a temporary `.docx` via `python-docx` and asserts `DefaultTextExtractor` returns the expected newline-joined paragraph text.
- **PDF OCR fallback control flow**:
  - Uses `monkeypatch` to:
    - Stub `pypdf.PdfReader` so the text layer is empty.
    - Stub `pdf2image.convert_from_path` and `pytesseract.image_to_string` so the basic OCR returns no text and the strong OCR path returns a known string.
  - Asserts that the extracted text contains the sentinel from the strong OCR path.

## Planned Integration Tests

The structure is in place and initial integration tests have been added.

### `tests/integration/test_pipeline.py`

- **Extraction pipeline integration** (`test_extract_documents_writes_corpus_and_stats`):
  - Uses a temporary root folder with a couple of supported files (.pdf, .jpg).
  - Sets `SMARTFILES_DATA_DIR` to a temporary directory.
  - Stubs the format-specific text extractor so that extraction always returns predictable text.
  - Calls `extract_documents(root_folder=..., recreate_text=True)`.
  - Asserts that:
    - The per-root corpus directory contains one `.txt` file per input document with the stubbed text.
    - The per-root stats directory contains exactly one `extraction_XXXX.txt` file with a `Summary:` section and `[OK]` entries.
- **Index-from-corpus integration** (`test_build_index_from_corpus_uses_saved_text`):
  - Uses `save_document_text` to populate the corpus for a synthetic root folder.
  - Stubs `get_default_embedding_model` and `get_default_vector_store` to lightweight fakes.
  - Calls `build_index_from_corpus(root_folder=..., recreate_index=True)`.
  - Asserts that the fake vector store's `add_documents` method is invoked with chunks whose combined text contains the sample corpus text.

## Planned End-to-End Tests

(Structure defined; tests to be added.) Examples to add here:

- **CLI search flow**:
  - Create a tiny dataset (e.g., 2–3 documents with distinctive phrases).
  - Run `smartfiles index <folder>` and then `smartfiles search "query"` via subprocess or Typer's CLI runner.
  - Assert that the top result filepaths and snippets match expectations.

## Running Tests

From `backend/`:

```bash
source .venv/bin/activate
pytest                # run all tests
pytest tests/unit     # run unit tests only
```

As integration and end-to-end suites are added, we can tag them with markers (e.g., `@pytest.mark.integration`, `@pytest.mark.e2e`) and update this document to show how to include or exclude them in day-to-day runs.
