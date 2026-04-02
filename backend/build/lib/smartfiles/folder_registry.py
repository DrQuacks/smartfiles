from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from smartfiles.config import get_data_dir


REGISTRY_FILENAME = "smartfiles_folders.json"


@dataclass
class FolderEntry:
    folder_name: str  # logical name used in `<name>_rawText`
    path: str         # absolute path to the root folder
    raw_text_dir_name: str
    last_indexed: Optional[str] = None
    last_commit: Optional[str] = None


def _get_registry_path() -> Path:
    """Return the path to the registry file inside the data directory.

    The data directory itself is determined by SMARTFILES_DATA_DIR (or
    defaults to ~/.smartfiles); we keep the registry alongside the
    database and *_rawText folders.
    """

    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / REGISTRY_FILENAME


def _get_old_registry_path() -> Path:
    """Previous location of the registry (repo root).

    This is kept for backward compatibility so existing installations
    with a registry at the old location continue to work. Once entries
    are saved again, they will be written to the new data-dir location.
    """

    # Repo root is two levels up from this file: backend/smartfiles/..
    return Path(__file__).resolve().parents[2] / REGISTRY_FILENAME


def _load_registry() -> List[FolderEntry]:
    path = _get_registry_path()
    if not path.exists():
        # Fallback to the old location if present.
        old_path = _get_old_registry_path()
        if not old_path.exists():
            return []
        path = old_path
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    entries: List[FolderEntry] = []
    for item in data:
        try:
            entries.append(
                FolderEntry(
                    folder_name=item["folder_name"],
                    path=item["path"],
                    raw_text_dir_name=item["raw_text_dir_name"],
                    last_indexed=item.get("last_indexed"),
                    last_commit=item.get("last_commit"),
                )
            )
        except Exception:
            continue
    return entries


def _save_registry(entries: List[FolderEntry]) -> None:
    path = _get_registry_path()
    data = [asdict(e) for e in entries]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_folders() -> List[FolderEntry]:
    """Return all known folder entries from the registry.

    This is a thin wrapper around the internal loader so other
    modules (such as the API layer) can inspect the registry
    without relying on private helpers.
    """

    return _load_registry()


def delete_folder_by_name(folder_name: str) -> bool:
    """Remove a folder entry from the registry by its logical name.

    Returns True if an entry was removed, False if no matching entry
    was found. This only affects the registry ordering and metadata;
    it does not delete any on-disk data or index contents.
    """

    entries = _load_registry()
    new_entries = [e for e in entries if e.folder_name != folder_name]
    if len(new_entries) == len(entries):
        return False
    _save_registry(new_entries)
    return True


def reorder_folders(order: List[str]) -> List[FolderEntry]:
    """Reorder registry entries to match the provided folder_name list.

    Any entries whose names are not present in ``order`` retain their
    relative ordering and are appended after the explicitly ordered
    entries. The updated list is persisted and returned.
    """

    entries = _load_registry()
    by_name = {e.folder_name: e for e in entries}

    ordered: List[FolderEntry] = []
    for name in order:
        entry = by_name.pop(name, None)
        if entry is not None:
            ordered.append(entry)

    # Append any remaining entries in their original order.
    for e in entries:
        if e.folder_name in by_name:
            ordered.append(e)

    _save_registry(ordered)
    return ordered


def ensure_folder_entry(root_folder: Path) -> FolderEntry:
    """Return or create a registry entry for the given root folder.

    If an entry already exists for this absolute path, it is returned.
    Otherwise a new name is chosen, avoiding collisions on folder_name
    by appending the parent folder name and, if needed, a numeric
    suffix.
    """

    root_folder = root_folder.expanduser().resolve()
    base_name = root_folder.name
    parent_name = root_folder.parent.name or "root"
    abs_path = str(root_folder)

    entries = _load_registry()

    # If we already know this exact path, reuse its entry.
    for e in entries:
        if e.path == abs_path:
            return e

    # Collect existing names to avoid collisions.
    existing_names = {e.folder_name for e in entries}

    candidate = base_name
    if candidate in existing_names:
        candidate = f"{base_name}-{parent_name}"

    # If still colliding, append a numeric suffix.
    suffix = 2
    while candidate in existing_names:
        candidate = f"{base_name}-{parent_name}-{suffix}"
        suffix += 1

    raw_text_dir_name = f"{candidate}_rawText"
    entry = FolderEntry(
        folder_name=candidate,
        path=abs_path,
        raw_text_dir_name=raw_text_dir_name,
    )
    entries.append(entry)
    _save_registry(entries)
    return entry


def update_folder_metadata(root_folder: Path, *, last_indexed: str, last_commit: str) -> None:
    """Update last_indexed / last_commit for the entry for root_folder."""

    root_folder = root_folder.expanduser().resolve()
    abs_path = str(root_folder)
    entries = _load_registry()
    changed = False
    for e in entries:
        if e.path == abs_path:
            e.last_indexed = last_indexed
            e.last_commit = last_commit
            changed = True
            break
    if changed:
        _save_registry(entries)


def get_raw_text_dir_name(root_folder: Path) -> str:
    """Return the raw_text_dir_name for this root folder (creating entry if needed)."""

    entry = ensure_folder_entry(root_folder)
    return entry.raw_text_dir_name
