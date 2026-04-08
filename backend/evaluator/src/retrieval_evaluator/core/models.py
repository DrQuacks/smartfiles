from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Hit:
    """Single retrieved document hit."""

    doc_id: str
    score: float


@dataclass
class RunConfig:
    """Configuration for a single benchmark run."""

    dataset: str
    split: str = "test"
    top_k: int = 10
    batch_size: int = 128
    backend_name: str = "unknown"
    tag: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a benchmark run, including metrics and metadata."""

    config: RunConfig
    timestamp: str
    duration_seconds: float
    metrics: Dict[str, Dict[str, float]]
    # e.g. {"ndcg": {"1": 0.6, "3": 0.7}, ...}
    backend_metadata: Dict[str, Any] = field(default_factory=dict)

    def metric_at(self, metric: str, k: int) -> Optional[float]:
        bucket = self.metrics.get(metric, {})
        return bucket.get(str(k))
