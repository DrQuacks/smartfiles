from __future__ import annotations

import pathlib
from typing import Iterable

from smartfiles.embeddings.embedding_model import get_default_embedding_model
from smartfiles.database.vector_store import get_default_vector_store
from smartfiles.database.text_store import reset_text_corpus, save_document_text
from smartfiles.ingestion.file_scanner import iter_files
from smartfiles.ingestion.text_extractor import get_default_extractor
from smartfiles.ingestion.chunker import chunk_document


def run_indexing_pipeline(*, root_folder: pathlib.Path, recreate: bool = False) -> None:
    """Run ingestion → embedding → storage for all supported files in a folder."""

    vector_store = get_default_vector_store(recreate=recreate)
    embedder = get_default_embedding_model()
    extractor = get_default_extractor()

    if recreate:
        reset_text_corpus()

    paths = list(iter_files(root_folder))
    if not paths:
        print(f"No supported files found under {root_folder}.")
        return

    print(f"Indexing {len(paths)} files under {root_folder}...")

    for path in paths:
        try:
            text = extractor.extract_text(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Failed to extract text from {path}: {exc}")
            continue

        if not text.strip():
            print(f"[WARN] No text extracted from {path}, skipping.")
            continue

        # Persist full raw text for inspection/validation.
        save_document_text(root_folder=root_folder, path=path, text=text)

        chunks = chunk_document(filepath=str(path), text=text)
        if not chunks:
            continue

        embeddings = embedder.embed_texts([c.text for c in chunks])
        vector_store.add_documents(chunks=chunks, embeddings=embeddings)

    print("Indexing complete.")
