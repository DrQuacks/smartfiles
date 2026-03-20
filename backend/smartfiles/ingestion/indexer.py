from __future__ import annotations

import pathlib
from typing import Iterable

from smartfiles.embeddings.embedding_model import get_default_embedding_model
from smartfiles.database.vector_store import get_default_vector_store
from smartfiles.database.text_store import (
    reset_text_corpus,
    save_document_text,
    iter_corpus_documents,
    get_next_stats_file_path,
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
        reset_text_corpus(root_folder)

    extractor = get_default_extractor()
    paths = list(iter_files(root_folder))
    if not paths:
        print(f"No supported files found under {root_folder}.")
        return

    print(f"Extracting text from {len(paths)} files under {root_folder}...")

    # Prepare a new stats file for this extraction run.
    from datetime import datetime
    import os
    import subprocess

    stats_path = get_next_stats_file_path(root_folder)

    # Try to capture the current git commit hash (best-effort).
    commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            commit = result.stdout.strip()
    except Exception:  # pragma: no cover - defensive
        pass

    timestamp = datetime.now().isoformat()
    data_dir = os.getenv("SMARTFILES_DATA_DIR", "~/.smartfiles (default)")

    header_lines = [
        "SmartFiles text extraction run",
        f"Timestamp: {timestamp}",
        f"Git commit: {commit}",
        f"Root folder: {root_folder}",
        f"SMARTFILES_DATA_DIR: {data_dir}",
        "---",
        "",
    ]
    stats_path.write_text("\n".join(header_lines), encoding="utf-8")

    for path in paths:
        try:
            text = extractor.extract_text(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Failed to extract text from {path}: {exc}")
            with stats_path.open("a", encoding="utf-8") as f:
                f.write(f"[ERROR] {path}\n")
                f.write(f"  error={exc}\n\n")
            continue

        if not text.strip():
            print(f"[WARN] No text extracted from {path}, skipping.")
            with stats_path.open("a", encoding="utf-8") as f:
                f.write(f"[WARN] {path}\n")
                f.write("  message=No text extracted; skipped\n\n")
            continue

        save_document_text(root_folder=root_folder, path=path, text=text)
        with stats_path.open("a", encoding="utf-8") as f:
            f.write(f"[OK] {path}\n")
            f.write(f"  extracted_chars={len(text)}\n\n")

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
