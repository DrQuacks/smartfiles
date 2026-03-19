from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator, Tuple

from smartfiles.config import get_data_dir

DEFAULT_TEXT_DIR = get_data_dir() / "corpus"


def reset_text_corpus() -> None:
    """Delete and recreate the text corpus directory."""
    if DEFAULT_TEXT_DIR.exists():
        shutil.rmtree(DEFAULT_TEXT_DIR)
    DEFAULT_TEXT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_text_corpus_dir() -> None:
    """Ensure the corpus directory exists without deleting it."""
    DEFAULT_TEXT_DIR.mkdir(parents=True, exist_ok=True)


def save_document_text(root_folder: Path, path: Path, text: str) -> Path:
    """Persist full raw text of a document as UTF-8 .txt.

    The directory structure under DEFAULT_TEXT_DIR mirrors the path
    relative to root_folder when possible. If the file is not inside
    root_folder, it is stored at the top level using its filename.

    Filenames keep their original extension and add ".txt" on top,
    e.g. "file.pdf" -> "file.pdf.txt". This allows us to reconstruct
    the original relative path later by stripping the trailing suffix.
    """
    ensure_text_corpus_dir()

    root_folder = root_folder.expanduser().resolve()
    path = path.expanduser().resolve()

    try:
        rel = path.relative_to(root_folder)
    except ValueError:
        # If the file is outside the root, just use the filename.
        rel = Path(path.name)

    # Store as "<rel>.txt" so we can drop the final .txt to get the
    # original relative path (including its extension).
    rel_txt = Path(str(rel) + ".txt")
    out_path = DEFAULT_TEXT_DIR / rel_txt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def iter_corpus_documents(root_folder: Path) -> Iterator[Tuple[Path, str]]:
    """Yield (original_path, text) for each document in the corpus.

    Assumes the corpus represents the most recent extraction run for
    the given root_folder and that files were written via
    save_document_text.
    """

    ensure_text_corpus_dir()
    root_folder = root_folder.expanduser().resolve()

    for txt_path in DEFAULT_TEXT_DIR.rglob("*.txt"):
        rel_txt = txt_path.relative_to(DEFAULT_TEXT_DIR)
        rel_str = str(rel_txt)
        if not rel_str.endswith(".txt"):
            continue
        rel_orig = Path(rel_str[:-4])  # strip trailing .txt
        original_path = (root_folder / rel_orig).expanduser().resolve()
        text = txt_path.read_text(encoding="utf-8")
        yield original_path, text
