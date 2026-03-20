from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterator, Tuple

from smartfiles.config import get_data_dir
from smartfiles.folder_registry import get_raw_text_dir_name


def _get_run_base_dir(root_folder: Path) -> Path:
    """Return the base directory for raw-text data for a root folder.

    Layout under the SmartFiles data dir (`SMARTFILES_DATA_DIR` or
    `~/.smartfiles` by default):

    `<DATA_DIR>/<folder_name>_rawText/{corpus,stats}/...`
    """

    data_dir = get_data_dir()
    folder_name = get_raw_text_dir_name(root_folder)
    return data_dir / folder_name


def get_corpus_dir(root_folder: Path) -> Path:
    return _get_run_base_dir(root_folder) / "corpus"


def get_stats_dir(root_folder: Path) -> Path:
    return _get_run_base_dir(root_folder) / "stats"


def reset_text_corpus(root_folder: Path) -> None:
    """Delete and recreate the text corpus directory for a root folder.

    This wipes the `corpus/` directory for the given root folder but
    leaves any existing stats files intact.
    """

    corpus_dir = get_corpus_dir(root_folder)
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # Ensure stats directory exists alongside the corpus.
    get_stats_dir(root_folder).mkdir(parents=True, exist_ok=True)


def ensure_text_corpus_dir(root_folder: Path) -> None:
    """Ensure the corpus directory (and stats root) exists."""

    get_corpus_dir(root_folder).mkdir(parents=True, exist_ok=True)
    get_stats_dir(root_folder).mkdir(parents=True, exist_ok=True)


def save_document_text(root_folder: Path, path: Path, text: str) -> Path:
    """Persist full raw text of a document as UTF-8 .txt.

    The directory structure under DEFAULT_TEXT_DIR mirrors the path
    relative to root_folder when possible. If the file is not inside
    root_folder, it is stored at the top level using its filename.

    Filenames keep their original extension and add ".txt" on top,
    e.g. "file.pdf" -> "file.pdf.txt". This allows us to reconstruct
    the original relative path later by stripping the trailing suffix.
    """
    ensure_text_corpus_dir(root_folder)

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
    out_path = get_corpus_dir(root_folder) / rel_txt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def iter_corpus_documents(root_folder: Path) -> Iterator[Tuple[Path, str]]:
    """Yield (original_path, text) for each document in the corpus.

    Assumes the corpus represents the most recent extraction run for
    the given root_folder and that files were written via
    save_document_text.
    """

    ensure_text_corpus_dir(root_folder)
    root_folder = root_folder.expanduser().resolve()

    corpus_dir = get_corpus_dir(root_folder)
    for txt_path in corpus_dir.rglob("*.txt"):
        rel_txt = txt_path.relative_to(corpus_dir)
        rel_str = str(rel_txt)
        if not rel_str.endswith(".txt"):
            continue
        rel_orig = Path(rel_str[:-4])  # strip trailing .txt
        original_path = (root_folder / rel_orig).expanduser().resolve()
        text = txt_path.read_text(encoding="utf-8")
        yield original_path, text


def get_next_stats_file_path(root_folder: Path) -> Path:
    """Return the path for the next stats file for this root folder.

    Files are named `extraction_0001.txt`, `extraction_0002.txt`, etc.,
    and live under the per-folder `stats/` directory.
    """

    stats_dir = get_stats_dir(root_folder)
    stats_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"extraction_(\d+)\.txt$")
    max_index = 0
    for entry in stats_dir.glob("extraction_*.txt"):
        match = pattern.match(entry.name)
        if match:
            try:
                idx = int(match.group(1))
            except ValueError:
                continue
            if idx > max_index:
                max_index = idx

    next_index = max_index + 1
    return stats_dir / f"extraction_{next_index:04d}.txt"
