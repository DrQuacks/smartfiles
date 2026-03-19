from __future__ import annotations

import pathlib
from typing import Iterable

from smartfiles.embeddings.embedding_model import get_default_embedding_model
from smartfiles.database.vector_store import get_default_vector_store
from smartfiles.database.text_store import (
    reset_text_corpus,
    save_document_text,
    iter_corpus_documents,
)
from smartfiles.ingestion.file_scanner import iter_files
from smartfiles.ingestion.text_extractor import get_default_extractor
from smartfiles.ingestion.chunker import chunk_document


def extract_documents(*, root_folder: pathlib.Path, recreate_text: bool = False) -> None:
    """Parse supported documents and write raw text files to the corpus.

    This stage is useful on its own to debug PDF parsing and OCR
    without running embeddings or touching the vector index.
    """

    if recreate_text:
        reset_text_corpus()

    extractor = get_default_extractor()
    paths = list(iter_files(root_folder))
    if not paths:
        print(f"No supported files found under {root_folder}.")
        return

    print(f"Extracting text from {len(paths)} files under {root_folder}...")

    for path in paths:
        try:
            text = extractor.extract_text(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Failed to extract text from {path}: {exc}")
            continue

        if not text.strip():
            print(f"[WARN] No text extracted from {path}, skipping.")
            continue

        save_document_text(root_folder=root_folder, path=path, text=text)

    print("Extraction complete.")


def build_index_from_corpus(*, root_folder: pathlib.Path, recreate_index: bool = False) -> None:
    """Chunk, embed, and index documents using the saved text corpus.

    Assumes `extract_documents` has already been run for the given
    root_folder, so that the corpus contains up-to-date `.txt` files.
    """

    vector_store = get_default_vector_store(recreate=recreate_index)
    embedder = get_default_embedding_model()

    any_docs = False
    for original_path, text in iter_corpus_documents(root_folder):
        any_docs = True
        if not text.strip():
            print(f"[WARN] Empty text in corpus for {original_path}, skipping.")
            continue

        chunks = chunk_document(filepath=str(original_path), text=text)
        if not chunks:
            continue

        embeddings = embedder.embed_texts([c.text for c in chunks])
        vector_store.add_documents(chunks=chunks, embeddings=embeddings)

    if not any_docs:
        print(
            "No documents found in the text corpus. "
            "Run 'smartfiles extract' first or use 'smartfiles index' for the full pipeline."
        )
    else:
        print("Index build complete.")


def run_indexing_pipeline(*, root_folder: pathlib.Path, recreate: bool = False) -> None:
    """Run extraction and index build as a single end-to-end pipeline."""

    # For a full run we recreate both the text corpus and the index
    # when requested.
    extract_documents(root_folder=root_folder, recreate_text=recreate)
    build_index_from_corpus(root_folder=root_folder, recreate_index=recreate)
