from __future__ import annotations

import shutil
from pathlib import Path

DEFAULT_TEXT_DIR = Path.home() / ".smartfiles" / "corpus"


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
    """
    ensure_text_corpus_dir()

    root_folder = root_folder.expanduser().resolve()
    path = path.expanduser().resolve()

    try:
        rel = path.relative_to(root_folder)
    except ValueError:
        # If the file is outside the root, just use the filename.
        rel = Path(path.name)

    out_path = DEFAULT_TEXT_DIR / rel
    out_path = out_path.with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
