from __future__ import annotations

"""Simple Streamlit dashboard for visualizing BEIR benchmark runs.

Usage (from backend/):

    source ../.venv/bin/activate
    pip install .[benchmark]
    streamlit run scripts/benchmark_dashboard.py

This reads the JSONL log produced by the BEIR benchmark harness at
SMARTFILES_DATA_DIR/benchmarks/beir/runs.jsonl and provides basic
filtering and comparison across models and datasets.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from smartfiles.config import get_data_dir


RUNS_REL_PATH = Path("benchmarks") / "beir" / "runs.jsonl"


@dataclass
class RunRecord:
    raw: Dict[str, Any]

    @property
    def dataset(self) -> str:
        return str(self.raw.get("dataset", ""))

    @property
    def split(self) -> str:
        return str(self.raw.get("split", ""))

    @property
    def tag(self) -> Optional[str]:
        val = self.raw.get("run_tag")
        return str(val) if val is not None else None

    @property
    def timestamp(self) -> str:
        return str(self.raw.get("timestamp", ""))

    @property
    def embedding_profile(self) -> Optional[str]:
        emb = self.raw.get("embedding") or {}
        val = emb.get("env_profile") or None
        return str(val) if val is not None else None

    @property
    def embedding_model(self) -> Optional[str]:
        emb = self.raw.get("embedding") or {}
        val = emb.get("model_name_or_path") or emb.get("env_model")
        return str(val) if val is not None else None

    @property
    def smartfiles_version(self) -> Optional[str]:
        val = self.raw.get("smartfiles_version")
        return str(val) if val is not None else None

    def metric_at(self, metric: str, k: int) -> Optional[float]:
        metrics = self.raw.get("metrics") or {}
        m = metrics.get(metric) or {}
        val = m.get(str(k))
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None


def load_runs() -> List[RunRecord]:
    data_dir = get_data_dir()
    runs_path = data_dir / RUNS_REL_PATH
    if not runs_path.exists():
        return []
    records: List[RunRecord] = []
    with runs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(RunRecord(raw=obj))
    return records


def to_dataframe(records: List[RunRecord]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in records:
        base: Dict[str, Any] = {
            "timestamp": r.timestamp,
            "dataset": r.dataset,
            "split": r.split,
            "tag": r.tag,
            "embedding_profile": r.embedding_profile,
            "embedding_model": r.embedding_model,
            "smartfiles_version": r.smartfiles_version,
        }
        # Include a few common metric@K columns for quick sorting.
        for metric in ("ndcg", "recall", "map", "precision"):
            for k in (1, 3, 5, 10):
                base[f"{metric}@{k}"] = r.metric_at(metric, k)
        rows.append(base)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Sort by timestamp descending by default.
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)
    return df


def main() -> None:
    st.set_page_config(page_title="SmartFiles BEIR Benchmarks", layout="wide")
    st.title("SmartFiles BEIR Benchmark Dashboard")

    records = load_runs()
    if not records:
        st.warning(
            "No benchmark runs found yet. Run `smartfiles benchmark-beir` "
            "or `python scripts/run_beir_matrix.py` first."
        )
        return

    df = to_dataframe(records)

    with st.sidebar:
        st.header("Filters")
        datasets = sorted(df["dataset"].dropna().unique().tolist())
        dataset_sel = st.multiselect("Dataset", datasets, default=datasets)

        profiles = sorted(
            [p for p in df["embedding_profile"].dropna().unique().tolist() if p]
        )
        profile_sel = st.multiselect("Embedding profile", profiles, default=profiles)

        tags = sorted([t for t in df["tag"].dropna().unique().tolist() if t])
        tag_sel = st.multiselect("Tag (optional)", tags, default=tags)

        metric = st.selectbox("Metric", ["ndcg", "recall", "map", "precision"], index=0)
        k = st.selectbox("K", [1, 3, 5, 10], index=3)

    filtered = df.copy()
    if dataset_sel:
        filtered = filtered[filtered["dataset"].isin(dataset_sel)]
    if profile_sel:
        filtered = filtered[filtered["embedding_profile"].isin(profile_sel)]
    if tag_sel:
        filtered = filtered[filtered["tag"].isin(tag_sel)]

    metric_col = f"{metric}@{k}"

    st.subheader(f"Runs ({metric.upper()}@{k})")
    if filtered.empty:
        st.info("No runs match the current filters.")
        return

    # Focus on runs that actually have this metric recorded.
    metric_df = filtered.dropna(subset=[metric_col])
    if metric_df.empty:
        st.info(
            f"No runs have recorded {metric.upper()}@{k} yet. "
            "Try a different metric/K or rerun benchmarks with the current code."
        )
        return

    # Summary by dataset/profile.
    group_cols = ["dataset", "embedding_profile", "embedding_model"]
    summary = (
        metric_df.groupby(group_cols)[metric_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    st.markdown("### Summary by dataset and profile")
    st.dataframe(summary, use_container_width=True)

    st.markdown("### Individual runs")
    display_cols = [
        "timestamp",
        "dataset",
        "split",
        "embedding_profile",
        "embedding_model",
        "tag",
        metric_col,
    ]
    st.dataframe(metric_df[display_cols], use_container_width=True)


if __name__ == "__main__":
    main()
