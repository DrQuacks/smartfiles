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
    save_chunk_text,
    get_corpus_dir,
)
from smartfiles.folder_registry import ensure_folder_entry, update_folder_metadata
from smartfiles.ingestion.file_scanner import iter_files
from smartfiles.ingestion.text_extractor import get_default_extractor
from smartfiles.ingestion.chunker import chunk_document


def extract_documents(*, root_folder: pathlib.Path, recreate_text: bool = False) -> None:
    """Parse supported documents and write raw text files to the corpus.

    This stage is useful on its own to debug PDF parsing and OCR
    without running embeddings or touching the vector index.
    """

    # Ensure there is a registry entry for this root folder and get
    # the canonical absolute path.
    entry = ensure_folder_entry(root_folder)
    root_folder = pathlib.Path(entry.path)

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

    start_time = datetime.now()
    timestamp = start_time.isoformat()
    data_dir = os.getenv("SMARTFILES_DATA_DIR", "~/.smartfiles (default)")

    header_lines = [
        "SmartFiles text extraction run",
        f"Timestamp: {timestamp}",
        f"Git commit: {commit}",
        f"Root folder: {root_folder}",
        f"SMARTFILES_DATA_DIR: {data_dir}",
        "",
    ]

    # Per-file stats and counters for summary.
    per_file_lines: list[str] = []
    ok_count = 0
    warn_count = 0
    skip_count = 0

    # Precompute the corpus directory so we can detect already-extracted
    # files and avoid rerunning heavy PDF/OCR work when not recreating.
    corpus_dir = get_corpus_dir(root_folder)

    for path in paths:
        # Skip zero-byte files before attempting extraction.
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:  # pragma: no cover - defensive
            print(f"[WARN] Could not stat {path}: {exc}")
            per_file_lines.append(f"[WARN] {path}\n")
            per_file_lines.append(f"  message=Could not stat file; skipped\n\n")
            warn_count += 1
            continue

        # If we're not recreating the corpus and a text file already
        # exists for this document, skip extraction and rely on the
        # existing corpus for downstream chunking/indexing.
        rel_target = None
        try:
            rel = path.expanduser().resolve().relative_to(root_folder)
            rel_target = corpus_dir / (str(rel) + ".txt")
        except ValueError:
            # If outside the root, we store at the top level using
            # just the filename.
            rel_target = corpus_dir / f"{path.name}.txt"

        if not recreate_text and rel_target is not None and rel_target.exists():
            print(f"[SKIP] {path} (already extracted; using existing corpus text)")
            per_file_lines.append(f"[SKIP] {path}\n")
            per_file_lines.append("  message=Already extracted; using existing corpus text\n\n")
            skip_count += 1
            continue

        if size_bytes == 0:
            print(f"[SKIP] {path} (0kb)")
            per_file_lines.append(f"[SKIP] {path}\n")
            per_file_lines.append("  message=0kb\n\n")
            skip_count += 1
            continue
        try:
            text = extractor.extract_text(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Failed to extract text from {path}: {exc}")
            per_file_lines.append(f"[ERROR] {path}\n")
            per_file_lines.append(f"  error={exc}\n\n")
            continue

        if not text.strip():
            print(f"[WARN] No text extracted from {path}, skipping.")
            per_file_lines.append(f"[WARN] {path}\n")
            per_file_lines.append("  message=No text extracted; skipped\n\n")
            warn_count += 1
            continue

        save_document_text(root_folder=root_folder, path=path, text=text)
        per_file_lines.append(f"[OK] {path}\n")
        per_file_lines.append(f"  extracted_chars={len(text)}\n\n")
        ok_count += 1

    # Record total duration and update registry metadata.
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Build summary and write stats file in one pass so that
    # summary counts appear near the top of the file.
    summary_lines = [
        "Summary:",
        f"  total_files={len(paths)}",
        f"  ok={ok_count}",
        f"  warn={warn_count}",
        f"  skip={skip_count}",
        "",
        "---",
        "",
    ]

    footer_lines = [
        "---",
        f"Total duration_seconds={duration:.3f}",
        "",
    ]

    all_lines = header_lines + summary_lines + per_file_lines + footer_lines
    stats_path.write_text("\n".join(all_lines), encoding="utf-8")

    update_folder_metadata(root_folder, last_indexed=end_time.isoformat(), last_commit=commit)

    print("Extraction complete.")


def build_index_from_corpus(
    *,
    root_folder: pathlib.Path,
    recreate_index: bool = False,
    save_chunks: bool = True,
    chunk_size: int = 500,
    overlap: int = 50,
) -> None:
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

        chunks = chunk_document(
            filepath=str(original_path),
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks:
            continue

        if save_chunks:
            for chunk in chunks:
                save_chunk_text(root_folder=root_folder, path=original_path, chunk_index=chunk.chunk_index, text=chunk.text)

        embeddings = embedder.embed_texts([c.text for c in chunks])
        vector_store.add_documents(chunks=chunks, embeddings=embeddings)

    if not any_docs:
        print(
            "No documents found in the text corpus. "
            "Run 'smartfiles extract' first or use 'smartfiles index' for the full pipeline."
        )
    else:
        print("Index build complete.")


def chunk_corpus_from_text(
    *,
    root_folder: pathlib.Path,
    save_chunks: bool = True,
    chunk_size: int = 500,
    overlap: int = 50,
) -> None:
    """Chunk documents using the saved text corpus, without embedding.

    This stage is useful for inspecting how documents are broken up
    into chunks before any embeddings or indexing are performed.
    """

    any_docs = False
    for original_path, text in iter_corpus_documents(root_folder):
        any_docs = True
        if not text.strip():
            print(f"[WARN] Empty text in corpus for {original_path}, skipping.")
            continue

        chunks = chunk_document(
            filepath=str(original_path),
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks:
            continue

        if save_chunks:
            for chunk in chunks:
                save_chunk_text(
                    root_folder=root_folder,
                    path=original_path,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                )

    if not any_docs:
        print(
            "No documents found in the text corpus. "
            "Run 'smartfiles extract' first or use 'smartfiles index' for the full pipeline."
        )
    else:
        print("Chunking complete.")


def run_indexing_pipeline(
    *,
    root_folder: pathlib.Path,
    recreate: bool = False,
    save_chunks: bool = True,
    chunk_size: int = 500,
    overlap: int = 50,
) -> None:
    """Run extraction and index build as a single end-to-end pipeline."""

    # For a full run we recreate both the text corpus and the index
    # when requested.
    extract_documents(root_folder=root_folder, recreate_text=recreate)
    build_index_from_corpus(
        root_folder=root_folder,
        recreate_index=recreate,
        save_chunks=save_chunks,
        chunk_size=chunk_size,
        overlap=overlap,
    )
