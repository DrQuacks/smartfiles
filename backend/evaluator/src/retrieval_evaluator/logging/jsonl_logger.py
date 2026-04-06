from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..core.models import RunResult


class JsonlRunLogger:
    """Append-only JSONL logger for benchmark runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, results: Iterable[RunResult]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for result in results:
                payload = asdict(result)
                f.write(json.dumps(payload) + "\n")
