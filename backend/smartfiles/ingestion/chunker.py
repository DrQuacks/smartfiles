from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass
class DocumentChunk:
    id: str
    filepath: str
    chunk_index: int
    text: str


def _iter_word_chunks(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    words = text.split()
    if not words:
        return
    i = 0
    n = len(words)
    while i < n:
        chunk_words = words[i : i + chunk_size]
        yield " ".join(chunk_words)
        if i + chunk_size >= n:
            break
        i += max(1, chunk_size - overlap)


def chunk_document(
    *,
    filepath: str,
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for idx, chunk_text in enumerate(_iter_word_chunks(text, chunk_size, overlap)):
        chunk_id = f"{filepath}::chunk-{idx}"
        chunks.append(
            DocumentChunk(
                id=chunk_id,
                filepath=filepath,
                chunk_index=idx,
                text=chunk_text,
            )
        )
    return chunks
