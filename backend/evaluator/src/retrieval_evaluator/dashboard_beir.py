from __future__ import annotations

"""Streamlit dashboard for visualizing retrieval-evaluator BEIR runs.

Usage (from backend/):

    source .venv/bin/activate
    pip install -e evaluator[dashboard]
    streamlit run evaluator/src/retrieval_evaluator/dashboard_beir.py

This reads the JSONL log produced by the evaluator's BEIR CLI
(default: ~/.retrieval_evaluator/beir_runs.jsonl) and provides basic
filtering and comparison across backends and datasets.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

import pandas as pd
import streamlit as st

from retrieval_evaluator.backends.smartfiles_backend import SmartFilesBackend
from retrieval_evaluator.core.beir_evaluator import evaluate_beir_run
from retrieval_evaluator.core.models import RunConfig
from retrieval_evaluator.datasets.beir import BeirDataset
from retrieval_evaluator.logging.jsonl_logger import JsonlRunLogger
from smartfiles.embeddings.embedding_model import (
    PROFILE_ENV_VAR,
    MODEL_ENV_VAR,
    list_supported_models,
)


DEFAULT_RUNS_PATH = Path.home() / ".retrieval_evaluator" / "beir_runs.jsonl"


@dataclass
class RunRecord:
    raw: Dict[str, Any]

    @property
    def dataset(self) -> str:
        cfg = self.raw.get("config") or {}
        return str(cfg.get("dataset", ""))

    @property
    def split(self) -> str:
        cfg = self.raw.get("config") or {}
        return str(cfg.get("split", ""))

    @property
    def tag(self) -> Optional[str]:
        cfg = self.raw.get("config") or {}
        val = cfg.get("tag")
        return str(val) if val is not None else None

    @property
    def backend_name(self) -> str:
        cfg = self.raw.get("config") or {}
        return str(cfg.get("backend_name", ""))

    @property
    def embedding_profile(self) -> Optional[str]:
        cfg = self.raw.get("config") or {}
        extra = cfg.get("extra_params") or {}
        val = extra.get("embedding_profile")
        return str(val) if val is not None else None

    @property
    def embedding_model_override(self) -> Optional[str]:
        cfg = self.raw.get("config") or {}
        extra = cfg.get("extra_params") or {}
        val = extra.get("embedding_model_override")
        return str(val) if val is not None else None

    @property
    def timestamp(self) -> str:
        # Prefer the top-level timestamp on RunResult; fall back to any
        # timestamp placed in backend_metadata for older runs.
        top_level = self.raw.get("timestamp")
        if top_level is not None:
            return str(top_level)
        meta = self.raw.get("backend_metadata") or {}
        val = meta.get("timestamp")
        return str(val) if val is not None else ""

    def metric_at(self, metric: str, k: int) -> Optional[float]:
        metrics = self.raw.get("metrics") or {}
        bucket = metrics.get(metric) or {}
        val = bucket.get(str(k))
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None


def load_runs(path: Path = DEFAULT_RUNS_PATH) -> List[RunRecord]:
    if not path.exists():
        return []
    records: List[RunRecord] = []
    with path.open("r", encoding="utf-8") as f:
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
            "dataset": r.dataset,
            "split": r.split,
            "tag": r.tag,
            "backend_name": r.backend_name,
            "embedding_profile": r.embedding_profile,
            "embedding_model": r.embedding_model_override,
            "timestamp": r.timestamp,
        }
        for metric in ("ndcg", "recall", "map", "precision"):
            for k in (1, 3, 5, 10):
                base[f"{metric}@{k}"] = r.metric_at(metric, k)
        rows.append(base)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.sort_values("timestamp", ascending=False)
    return df


def main() -> None:
    st.set_page_config(page_title="Retrieval Evaluator BEIR Runs", layout="wide")
    st.title("Retrieval Evaluator BEIR Dashboard")

    runs_path = DEFAULT_RUNS_PATH
    st.caption(f"Reading runs from: {runs_path}")

    if "is_running_benchmark" not in st.session_state:
        st.session_state["is_running_benchmark"] = False

    with st.expander("Run new BEIR benchmark", expanded=False):
        dataset_name = st.text_input("Dataset name", value="scifact", key="dataset_name")
        dataset_dir = st.text_input(
            "Dataset directory",
            value="",
            help="Path to the BEIR dataset folder (containing corpus.jsonl, queries.jsonl, qrels/)",
            key="dataset_dir",
        )
        split = st.text_input("Split", value="test", key="split")
        top_k = st.number_input("Top K", min_value=1, max_value=1000, value=10, step=1, key="top_k")
        batch_size = st.number_input("Batch size", min_value=1, max_value=2048, value=128, step=1, key="batch_size")
        tag = st.text_input("Tag (optional)", value="", key="run_tag")

        st.markdown("**Embedding models**")
        supported = list_supported_models()
        profile_keys = [m.key for m in supported]
        profile_labels = {m.key: f"{m.key} — {m.description}" for m in supported}
        current_profile = os.getenv(PROFILE_ENV_VAR, profile_keys[0] if profile_keys else "")
        default_profiles = [current_profile] if current_profile in profile_keys else profile_keys
        selected_profiles = st.multiselect(
            "Profiles (SMARTFILES_EMBEDDING_PROFILE)",
            options=profile_keys,
            default=default_profiles,
            format_func=lambda k: profile_labels.get(k, k),
            key="embedding_profiles",
        )

        custom_model = st.text_input(
            "Override model id/path (SMARTFILES_EMBEDDING_MODEL)",
            value=os.getenv(MODEL_ENV_VAR, ""),
            help="Optional: Hugging Face model id or local SentenceTransformers path. Takes precedence over profile.",
            key="embedding_model_override",
        )

        run_clicked = st.button(
            "Run benchmark(s)",
            key="run_benchmark",
            disabled=st.session_state.get("is_running_benchmark", False),
        )

        if run_clicked and not st.session_state.get("is_running_benchmark", False):
            data_path = Path(dataset_dir).expanduser()
            if not data_path.exists() or not data_path.is_dir():
                st.error("Dataset directory does not exist or is not a directory.")
            elif not selected_profiles:
                st.error("Please select at least one embedding profile to run.")
            else:
                st.session_state["is_running_benchmark"] = True
                try:
                    with st.spinner("Running BEIR evaluation(s) with SmartFiles backend..."):
                        dataset = BeirDataset(name=dataset_name, data_dir=str(data_path))
                        corpus, queries, qrels = dataset.load(split=split)

                        results_to_log = []

                        for profile_choice in selected_profiles:
                            # Configure embedding model environment for this run.
                            os.environ[PROFILE_ENV_VAR] = profile_choice
                            if custom_model.strip():
                                os.environ[MODEL_ENV_VAR] = custom_model.strip()
                            elif MODEL_ENV_VAR in os.environ:
                                os.environ.pop(MODEL_ENV_VAR)

                            backend = SmartFilesBackend()
                            config = RunConfig(
                                dataset=dataset_name,
                                split=split,
                                top_k=int(top_k),
                                batch_size=int(batch_size),
                                backend_name=backend.name,
                                tag=tag or None,
                                extra_params={
                                    "embedding_profile": profile_choice,
                                    "embedding_model_override": custom_model.strip() or None,
                                },
                            )

                            result = evaluate_beir_run(
                                backend=backend,
                                corpus=corpus,
                                queries=queries,
                                qrels=qrels,
                                config=config,
                            )

                            results_to_log.append(result)

                        if results_to_log:
                            logger = JsonlRunLogger(runs_path)
                            logger.append(results_to_log)

                    st.success(f"Completed and logged {len(selected_profiles)} benchmark run(s).")
                finally:
                    st.session_state["is_running_benchmark"] = False

    records = load_runs(runs_path)
    if not records:
        st.warning(
            "No evaluator runs found yet. Run the evaluator BEIR CLI first "
            "(e.g. via retrieval_evaluator.cli_smartfiles_beir)."
        )
        return

    df = to_dataframe(records)

    with st.sidebar:
        st.header("Filters")
        datasets = sorted(df["dataset"].dropna().unique().tolist())
        dataset_sel = st.multiselect("Dataset", datasets, default=datasets)

        profiles = sorted([p for p in df["embedding_profile"].dropna().unique().tolist() if p])
        profile_sel = st.multiselect("Embedding profile", profiles, default=profiles)

        backends = sorted(df["backend_name"].dropna().unique().tolist())
        backend_sel = st.multiselect("Backend", backends, default=backends)

        tags = sorted([t for t in df["tag"].dropna().unique().tolist() if t])
        tag_sel = st.multiselect("Tag (optional)", tags, default=tags)

        metric = st.selectbox("Metric", ["ndcg", "recall", "map", "precision"], index=0)
        k = st.selectbox("K", [1, 3, 5, 10], index=3)

    filtered = df.copy()
    if dataset_sel:
        filtered = filtered[filtered["dataset"].isin(dataset_sel)]
    if profile_sel:
        filtered = filtered[filtered["embedding_profile"].isin(profile_sel)]
    if backend_sel:
        filtered = filtered[filtered["backend_name"].isin(backend_sel)]
    if tag_sel:
        filtered = filtered[filtered["tag"].isin(tag_sel)]

    metric_col = f"{metric}@{k}"

    st.subheader(f"Runs ({metric.upper()}@{k})")
    if filtered.empty:
        st.info("No runs match the current filters.")
        return

    metric_df = filtered.dropna(subset=[metric_col])
    if metric_df.empty:
        st.info(
            f"No runs have recorded {metric.upper()}@{k} yet. "
            "Try a different metric/K or rerun benchmarks."
        )
        return

    group_cols = ["dataset", "embedding_profile", "backend_name"]
    summary = (
        metric_df.groupby(group_cols)[metric_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    st.markdown("### Summary by dataset and backend")
    st.dataframe(summary, use_container_width=True)

    st.markdown("### Individual runs")
    display_cols = [
        "dataset",
        "split",
        "embedding_profile",
        "embedding_model",
        "backend_name",
        "tag",
        "timestamp",
        metric_col,
    ]
    st.dataframe(metric_df[display_cols], use_container_width=True)


if __name__ == "__main__":  # pragma: no cover
    main()
