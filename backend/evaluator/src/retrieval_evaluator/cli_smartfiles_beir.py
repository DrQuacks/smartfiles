from __future__ import annotations

from .cli_beir import run_with_backend
from .backends.smartfiles_backend import SmartFilesBackend


def main() -> None:
    backend = SmartFilesBackend()
    run_with_backend(backend)


if __name__ == "__main__":  # pragma: no cover
    main()
