from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple


PAGE_MARK_PREFIX = "[[[SMARTFILES_PAGE "
PAGE_MARK_SUFFIX = "]]]"


@dataclass
class DocumentChunk:
    id: str
    filepath: str
    chunk_index: int
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


def _tokenize_with_pages(text: str) -> list[Tuple[str, int]]:
    """Split text into (word, page) pairs using page markers when present.

    For PDF-derived corpus text, pages are delimited with lightweight
    markers of the form `[[[SMARTFILES_PAGE N]]]`. For other documents,
    or if markers are absent, all content is treated as page 1.
    """

    words_with_pages: list[Tuple[str, int]] = []
    current_page = 1
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(PAGE_MARK_PREFIX) and stripped.endswith(PAGE_MARK_SUFFIX):
            num_str = stripped[len(PAGE_MARK_PREFIX) : -len(PAGE_MARK_SUFFIX)]
            try:
                current_page = int(num_str)
            except ValueError:
                # Ignore malformed markers and keep the current page.
                pass
            continue
        for word in line.split():
            words_with_pages.append((word, current_page))
    return words_with_pages


def _iter_word_chunks_with_pages(text: str, chunk_size: int, overlap: int) -> Iterator[Tuple[str, Optional[int], Optional[int]]]:
    words_with_pages = _tokenize_with_pages(text)
    if not words_with_pages:
        return

    i = 0
    n = len(words_with_pages)
    while i < n:
        slice_items = words_with_pages[i : i + chunk_size]
        chunk_words = [w for (w, _p) in slice_items]
        pages = [p for (_w, p) in slice_items]
        page_start = min(pages) if pages else None
        page_end = max(pages) if pages else None
        yield " ".join(chunk_words), page_start, page_end
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
    for idx, (chunk_text, page_start, page_end) in enumerate(
        _iter_word_chunks_with_pages(text, chunk_size, overlap)
    ):
        chunk_id = f"{filepath}::chunk-{idx}"
        chunks.append(
            DocumentChunk(
                id=chunk_id,
                filepath=filepath,
                chunk_index=idx,
                text=chunk_text,
                page_start=page_start,
                page_end=page_end,
            )
        )
    return chunks
