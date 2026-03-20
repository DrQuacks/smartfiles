import pathlib
from collections.abc import Iterable
from typing import Iterator

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".docx"}


def iter_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield supported files under root recursively."""
    root = root.expanduser().resolve()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def list_files(root: pathlib.Path) -> list[pathlib.Path]:
    return list(iter_files(root))
