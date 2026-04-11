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
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import altair as alt
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
ERRORS_PATH = Path.home() / ".retrieval_evaluator" / "beir_errors.jsonl"


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
        if val is None:
            meta = self.raw.get("backend_metadata") or {}
            val = meta.get("embedding_profile")
        return str(val) if val is not None else None

    @property
    def embedding_model_override(self) -> Optional[str]:
        cfg = self.raw.get("config") or {}
        extra = cfg.get("extra_params") or {}
        val = extra.get("embedding_model_override")
        if val is None:
            meta = self.raw.get("backend_metadata") or {}
            val = meta.get("embedding_model_override")
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

    @property
    def duration_seconds(self) -> Optional[float]:
        # Older runs may not have a duration recorded; in that case,
        # leave this as None so the UI can render it as blank.
        val = self.raw.get("duration_seconds")
        try:
            return float(val) if val is not None else None
        except Exception:
            return None

    @property
    def duration_hms(self) -> Optional[str]:
        """Return duration as H:MM:SS.mmm or None if missing."""

        total = self.duration_seconds
        if total is None:
            return None
        if total < 0:
            return None
        millis = int(round(total * 1000))
        seconds, ms = divmod(millis, 1000)
        minutes, sec = divmod(seconds, 60)
        hours, min_ = divmod(minutes, 60)
        return f"{hours:d}:{min_:02d}:{sec:02d}.{ms:03d}"

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
            # For legacy runs without duration, use NaN/empty string so
            # the table doesn't show a literal "None".
            "duration_seconds": r.duration_seconds if r.duration_seconds is not None else float("nan"),
            "duration": r.duration_hms or "",
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

    # Show any recently logged errors so that failures from previous
    # runs are visible even after a reload.
    with st.expander("Recent benchmark errors", expanded=False):
        if ERRORS_PATH.exists():
            try:
                rows: List[Dict[str, Any]] = []
                with ERRORS_PATH.open("r", encoding="utf-8") as ef:
                    for line in ef:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rows.append(obj)
                if rows:
                    # Show the most recent errors first.
                    rows = rows[-50:]
                    df_err = pd.DataFrame(rows)
                    if "timestamp" in df_err.columns:
                        df_err = df_err.sort_values("timestamp", ascending=False)
                    st.dataframe(
                        df_err[
                            [
                                c
                                for c in [
                                    "timestamp",
                                    "dataset",
                                    "split",
                                    "embedding_profile",
                                    "embedding_model_override",
                                    "backend_name",
                                    "error",
                                ]
                                if c in df_err.columns
                            ]
                        ],
                        use_container_width=True,
                    )
                else:
                    st.caption("No errors logged yet.")
            except Exception:
                st.caption("Failed to read error log.")
        else:
            st.caption("No errors logged yet.")

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

                        total_profiles = len(selected_profiles)
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0.0)
                        logger = JsonlRunLogger(runs_path)

                        successful = 0

                        for idx, profile_choice in enumerate(selected_profiles, start=1):
                            status_placeholder.write(
                                f"Running profile {profile_choice} ({idx}/{total_profiles})..."
                            )

                            # Configure embedding model environment for this run.
                            os.environ[PROFILE_ENV_VAR] = profile_choice
                            if custom_model.strip():
                                os.environ[MODEL_ENV_VAR] = custom_model.strip()
                            elif MODEL_ENV_VAR in os.environ:
                                os.environ.pop(MODEL_ENV_VAR)

                            try:
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

                                # Log each successful run immediately so partial
                                # successes are preserved even if a later profile fails.
                                logger.append([result])
                                successful += 1
                            except Exception as exc:  # noqa: BLE001
                                # Surface the error in the UI and also persist a
                                # structured record so failures are debuggable later.
                                st.error(
                                    f"Error while running profile '{profile_choice}': {exc}"
                                )

                                try:
                                    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
                                    error_record = {
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "dataset": dataset_name,
                                        "split": split,
                                        "embedding_profile": profile_choice,
                                        "embedding_model_override": custom_model.strip() or None,
                                        "backend_name": getattr(backend, "name", None),
                                        "error": str(exc),
                                        "traceback": traceback.format_exc(),
                                    }
                                    with ERRORS_PATH.open("a", encoding="utf-8") as ef:
                                        ef.write(json.dumps(error_record) + "\n")
                                except Exception:
                                    # If logging the error itself fails, don't crash the UI.
                                    pass

                            progress_bar.progress(idx / total_profiles)

                        status_placeholder.write(
                            f"Completed {successful} successful run(s) out of {total_profiles} profile(s)."
                        )

                    if successful:
                        st.success(
                            f"Completed and logged {successful} benchmark run(s). "
                            "Check above for any profiles that failed."
                        )
                    else:
                        st.error("All selected profiles failed. See errors above for details.")
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

        metric_options = ["ndcg", "recall", "map", "precision"]
        metric = st.selectbox("Metric", metric_options, index=0)
        metric_help = {
            "ndcg": "NDCG@K ∈ [0,1]: higher is better; measures ranking quality with more weight on top-ranked relevant docs.",
            "recall": "Recall@K ∈ [0,1]: higher is better; fraction of all relevant docs retrieved in top-K.",
            "map": "MAP@K ∈ [0,1]: higher is better; average precision across ranks and queries.",
            "precision": "Precision@K ∈ [0,1]: higher is better; fraction of top-K results that are relevant.",
        }
        st.caption(metric_help.get(metric, ""))
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
        "duration",
        metric_col,
    ]
    st.dataframe(metric_df[display_cols], use_container_width=True)

    st.markdown("### Visualization")
    st.caption("Pick any columns for X and Y to explore trade-offs (e.g. duration vs NDCG@10).")

    if not metric_df.empty:
        numeric_cols = sorted(metric_df.select_dtypes(include=["number"]).columns.tolist())
        all_cols = metric_df.columns.tolist()

        if not numeric_cols:
            st.info("No numeric columns available for plotting.")
        else:
            default_x = "duration_seconds" if "duration_seconds" in numeric_cols else numeric_cols[0]
            default_y = metric_col if metric_col in numeric_cols else numeric_cols[0]

            x_axis = st.selectbox("X axis", all_cols, index=all_cols.index(default_x) if default_x in all_cols else 0)
            y_axis = st.selectbox("Y axis", numeric_cols, index=numeric_cols.index(default_y) if default_y in numeric_cols else 0)

            x_is_numeric = x_axis in numeric_cols

            chart = (
                alt.Chart(metric_df)
                .mark_circle(size=80, opacity=0.8)
                .encode(
                    x=alt.X(x_axis, type="quantitative" if x_is_numeric else "nominal"),
                    y=alt.Y(y_axis, type="quantitative"),
                    color=alt.Color("embedding_profile:N", title="Profile"),
                    tooltip=[
                        "dataset",
                        "embedding_profile",
                        "embedding_model",
                        "backend_name",
                        "tag",
                        "timestamp",
                        "duration",
                        y_axis,
                    ],
                )
                .interactive()
            )

            st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":  # pragma: no cover
    main()
