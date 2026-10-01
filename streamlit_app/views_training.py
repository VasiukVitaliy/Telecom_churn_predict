"""Baseline / Optuna training controls, job monitor and MLflow model registry (REST only)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.api_client import ApiClient, ApiError

MODELS = ["lightgbm", "xgboost", "random_forest"]
SAMPLING = ["class_weight", "smote", "random_over", "none"]
NUM_IMPUTE = ["median", "mean", "constant", "iterative"]
CAT_IMPUTE = ["most_frequent", "constant"]
OPTIMIZE = ["pr_auc", "roc_auc", "f1"]
PARAM_SPACE_EXAMPLE = '{\n  "max_depth": {"type": "int", "low": 3, "high": 8},\n  "learning_rate": {"type": "float", "low": 0.02, "high": 0.2, "log": true}\n}'


def _fmt_ts(ts: float | None) -> str:
    return "—" if not ts else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _training_form(client: ApiClient) -> None:
    try:
        datasets = client.list_datasets()
    except ApiError as exc:
        st.error(str(exc))
        return
    if not datasets:
        st.info("Create a dataset in the EDA tab first.")
        return
    with st.form("train_form"):
        c1, c2, c3 = st.columns(3)
        dataset_id = c1.selectbox("Dataset", [d["dataset_id"] for d in datasets])
        model_name = c2.selectbox("Algorithm", MODELS)
        mode = c3.radio("Mode", ["baseline", "optuna"], horizontal=True)
        target = st.text_input("Target column (blank = auto-detect)")
        c1, c2, c3, c4 = st.columns(4)
        sampling = c1.selectbox("Imbalance handling", SAMPLING)
        ratio = c2.slider("Sampling ratio (minority/majority)", 0.1, 1.0, 0.6, 0.05)
        test_size = c3.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        folds = c4.slider("CV folds", 2, 10, 5)
        c1, c2, c3, c4 = st.columns(4)
        num_imp = c1.selectbox("Numeric imputation", NUM_IMPUTE)
        cat_imp = c2.selectbox("Categorical imputation", CAT_IMPUTE)
        scale = c3.checkbox("Scale numeric features", value=False)
        threshold = c4.slider("Decision threshold", 0.05, 0.95, 0.5, 0.01)
        with st.expander("Optuna settings (used when mode = optuna)"):
            c1, c2, c3, c4 = st.columns(4)
            trials = c1.number_input("Trials", 1, 500, 25)
            timeout = c2.number_input("Timeout seconds (0 = none)", 0, 86_400, 0, step=30)
            metric = c3.selectbox("Optimise metric", OPTIMIZE)
            n_jobs = c4.number_input("Parallel trials", 1, 16, 1)
            space_raw = st.text_area("Search-space overrides (JSON)", value="", placeholder=PARAM_SPACE_EXAMPLE, height=130)
        c1, c2, c3 = st.columns(3)
        run_name = c1.text_input("Run name (optional)")
        register = c2.checkbox("Register in model registry", value=True)
        seed = c3.number_input("Random state", value=42, step=1)
        submitted = st.form_submit_button("🚀 Start training job", type="primary")
    if not submitted:
        return
    try:
        space = json.loads(space_raw) if space_raw.strip() else None
    except json.JSONDecodeError as exc:
        st.error(f"Search-space JSON is invalid: {exc}")
        return
    config: dict[str, Any] = {
        "model_name": model_name, "mode": mode, "target_column": target.strip() or None,
        "test_size": test_size, "cv_folds": folds, "sampling": sampling, "sampling_ratio": ratio,
        "numeric_imputation": num_imp, "categorical_imputation": cat_imp, "scale_numeric": scale,
        "decision_threshold": threshold, "n_trials": int(trials), "timeout_seconds": int(timeout) or None,
        "optimize_metric": metric, "optuna_n_jobs": int(n_jobs), "param_space": space,
        "run_name": run_name.strip() or None, "register": register, "random_state": int(seed),
    }
    try:
        job = client.start_training(dataset_id, config)
        st.session_state["last_job"] = job["job_id"]
        st.success(f"Job {job['job_id']} {job['status']}. Follow it below.")
    except ApiError as exc:
        st.error(str(exc))
        if isinstance(exc.payload, dict) and exc.payload.get("report"):
            for err in exc.payload["report"].get("errors", []):
                st.write(f"• {err}")


@st.fragment(run_every=3)
def _jobs_panel(client: ApiClient) -> None:
    """Auto-refreshing job table (does not block the rest of the UI)."""
    try:
        jobs = client.list_jobs()
    except ApiError as exc:
        st.error(str(exc))
        return
    if not jobs:
        st.caption("No training jobs yet.")
        return
    rows = []
    for j in jobs:
        res = j.get("result") or {}
        rows.append({
            "job": j["job_id"], "status": j["status"], "model": j["config"]["model_name"], "mode": j["config"]["mode"],
            "stage": (j.get("progress") or {}).get("stage"),
            "trial": (j.get("progress") or {}).get("trial"),
            "version": res.get("model_version"),
            "test ROC-AUC": (res.get("test_metrics") or {}).get("roc_auc"),
            "test PR-AUC": (res.get("test_metrics") or {}).get("pr_auc"),
            "created": j["created_at"][:19].replace("T", " "), "error": j.get("error"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    chosen = st.selectbox("Inspect job", [j["job_id"] for j in jobs],
                          index=0, key="inspect_job")
    job = next(j for j in jobs if j["job_id"] == chosen)
    if job["status"] in ("queued", "running"):
        st.progress(0.35 if job["status"] == "queued" else 0.65, text=f"{job['status']}: {job.get('progress')}")
    elif job["status"] == "failed":
        st.error(job["error"])
    else:
        _job_result(job["result"])


def _job_result(res: dict[str, Any]) -> None:
    m = res["test_metrics"]
    cols = st.columns(7)
    for col, key in zip(cols, ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "brier_score"]):
        col.metric(key.replace("_", " ").upper(), f"{m[key]:.3f}")
    left, right = st.columns(2)
    with left:
        cm = res["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=cm, x=["pred 0", "pred 1"], y=["true 0", "true 1"], text=cm,
                                   texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(title="Confusion matrix (test set)", yaxis_autorange="reversed", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cv = pd.DataFrame(res["cv_metrics"]).T.reset_index(names="metric")
        fig = px.bar(cv, x="metric", y="mean", error_y="std", title="Cross-validation (mean ± std)")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{res['duration_s']:.1f}s · train={res['n_train']:,} test={res['n_test']:,} · "
               f"run {res['run_id']} · trials={res['n_trials']}")
    if res["best_params"]:
        st.json(res["best_params"], expanded=False)
    for warning in res.get("warnings", []):
        st.warning(warning)


def _registry(client: ApiClient) -> None:
    try:
        versions = client.model_versions()
    except ApiError as exc:
        st.error(str(exc))
        return
    if not versions:
        st.info("No registered model versions yet.")
        return
    rows = [{
        "version": v["version"], "aliases": ", ".join(f"@{a}" for a in v["aliases"]) or "—",
        "model": v["tags"].get("model_name"), "mode": v["tags"].get("mode"),
        "test ROC-AUC": v["metrics"].get("test_roc_auc"), "test PR-AUC": v["metrics"].get("test_pr_auc"),
        "test F1": v["metrics"].get("test_f1"), "created": _fmt_ts(v["created_at"]), "run": v["run_id"][:8],
    } for v in versions]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    nums = [v["version"] for v in versions]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Alias management**")
        version = st.selectbox("Version", nums, key="alias_version")
        alias = st.text_input("Alias", value="challenger", key="alias_name")
        a, b = st.columns(2)
        if a.button("Set alias"):
            _call(lambda: client.set_alias(version, alias), f"@{alias} → v{version}")
        if b.button("⭐ Promote to @champion"):
            _call(lambda: client.promote(version), f"@champion → v{version}")
    with c2:
        st.markdown("**Delete alias**")
        existing = sorted({a for v in versions for a in v["aliases"]})
        if existing:
            target = st.selectbox("Alias to delete", existing, key="alias_del")
            if st.button("Delete alias"):
                _call(lambda: client.delete_alias(target), f"Deleted @{target}")
        else:
            st.caption("No aliases defined.")
    with c3:
        st.markdown("**Zero-downtime hot reload**")
        mode = st.radio("Reload by", ["alias", "version"], horizontal=True)
        if mode == "alias":
            alias_r = st.text_input("Alias", value="champion", key="reload_alias")
            if st.button("🔄 Reload model"):
                _call(lambda: client.reload(alias=alias_r), f"Reloaded @{alias_r}")
        else:
            ver_r = st.selectbox("Version", nums, key="reload_version")
            if st.button("🔄 Reload version"):
                _call(lambda: client.reload(version=ver_r), f"Reloaded v{ver_r}")
    try:
        info = client.model_info()
        with st.expander("Serving model details & reload history"):
            st.json({k: info[k] for k in ("name", "version", "alias", "run_id", "threshold")})
            if info["reload_history"]:
                hist = pd.DataFrame(info["reload_history"])
                hist["at"] = hist["at"].map(_fmt_ts)
                st.dataframe(hist, use_container_width=True, hide_index=True)
            if info["feature_importance"]:
                imp = pd.DataFrame(info["feature_importance"]).head(20)
                st.plotly_chart(px.bar(imp[::-1], x=imp.columns[1], y=imp.columns[0], orientation="h",
                                       title="Top feature importances"), use_container_width=True)
    except ApiError:
        st.caption("No model is serving yet.")


def _call(fn: Any, success: str) -> None:
    try:
        result = fn()
        suffix = f" ({result['duration_ms']} ms)" if isinstance(result, dict) and "duration_ms" in result else ""
        st.success(success + suffix)
    except ApiError as exc:
        st.error(str(exc))


def _runs(client: ApiClient) -> None:
    try:
        runs = client.model_runs(100)
    except ApiError as exc:
        st.error(str(exc))
        return
    if not runs:
        st.caption("No runs recorded.")
        return
    rows = [{"run": r["run_id"][:8], "name": r["name"], "status": r["status"], "started": _fmt_ts(r["start_time"]),
             "model": r["params"].get("model_name"), "mode": r["params"].get("mode"),
             "test ROC-AUC": r["metrics"].get("test_roc_auc"), "test PR-AUC": r["metrics"].get("test_pr_auc"),
             "duration_s": r["metrics"].get("duration_s")} for r in runs]
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if frame["test PR-AUC"].notna().any():
        st.plotly_chart(px.scatter(frame, x="started", y="test PR-AUC", color="model", symbol="mode",
                                   title="Run quality over time"), use_container_width=True)


def render(client: ApiClient) -> None:
    """Render the Training & Registry tab.

    Args:
        client: API client.
    """
    sub = st.tabs(["Train", "Jobs", "Model registry", "MLflow runs"])
    with sub[0]:
        _training_form(client)
    with sub[1]:
        _jobs_panel(client)
    with sub[2]:
        _registry(client)
    with sub[3]:
        _runs(client)
