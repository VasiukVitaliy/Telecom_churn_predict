"""EDA, data-quality and interactive PCA views (REST only)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.api_client import ApiClient, ApiError


def _show_error(exc: ApiError) -> None:
    st.error(f"{exc}")
    report = exc.payload.get("report") if isinstance(exc.payload, dict) else None
    if report:
        for err in report.get("errors", []):
            st.write(f"• {err}")


def _dataset_picker(client: ApiClient) -> str | None:
    """Dataset management widgets; returns the selected dataset id."""
    with st.expander("➕ Add data", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Upload a real dataset** (CSV / Parquet)")
            upload = st.file_uploader("File", type=["csv", "tsv", "parquet"], key="eda_upload")
            if upload is not None and st.button("Upload", key="eda_upload_btn"):
                try:
                    info = client.upload_dataset(upload.name, upload.getvalue())
                    st.success(f"Stored {info['dataset_id']} ({info['n_rows']} rows × {info['n_columns']} cols)")
                except ApiError as exc:
                    _show_error(exc)
        with c2:
            st.markdown("**Generate synthetic subscribers**")
            n = st.slider("Rows", 500, 100_000, 20_000, 500, key="syn_n")
            rate = st.slider("Target churn rate", 0.05, 0.5, 0.20, 0.01, key="syn_rate")
            miss = st.slider("Missing rate", 0.0, 0.5, 0.0, 0.01, key="syn_miss")
            seed = st.number_input("Random state", value=42, step=1, key="syn_seed")
            if st.button("Generate", key="syn_btn"):
                try:
                    with st.spinner("Generating…"):
                        info = client.generate_synthetic(n_samples=n, target_churn_rate=rate,
                                                         missing_rate=miss, random_state=int(seed))
                    st.success(f"Stored {info['dataset_id']} ({info['n_rows']} × {info['n_columns']})")
                except ApiError as exc:
                    _show_error(exc)
    try:
        datasets = client.list_datasets()
    except ApiError as exc:
        _show_error(exc)
        return None
    if not datasets:
        st.info("No datasets yet — upload a file or generate synthetic data.")
        return None
    ids = [d["dataset_id"] for d in datasets]
    return st.selectbox("Dataset", ids, key="dataset_id")


def _render_quality(client: ApiClient, dataset_id: str, target: str | None) -> None:
    """Schema inference + quality gates."""
    if st.button("Run quality gates", key="validate_btn"):
        try:
            res = client.validate_dataset(dataset_id, target)
        except ApiError as exc:
            _show_error(exc)
            return
        report = res["report"]
        (st.success if report["passed"] else st.error)(
            "All quality gates passed" if report["passed"] else "Quality gates FAILED")
        for err in report["errors"]:
            st.error(err)
        for warn in report["warnings"]:
            st.warning(warn)
        cols = pd.DataFrame(res["schema"]["columns"]).T if isinstance(res["schema"]["columns"], dict) \
            else pd.DataFrame(res["schema"]["columns"])
        keep = [c for c in ("name", "kind", "dtype", "n_unique", "missing_pct", "min_value", "max_value") if c in cols]
        st.dataframe(cols[keep], use_container_width=True, hide_index=True)


def _render_summary(client: ApiClient, dataset_id: str, target: str | None) -> dict[str, Any] | None:
    try:
        eda = client.eda(dataset_id, target)
    except ApiError as exc:
        _show_error(exc)
        return None
    m = st.columns(5)
    m[0].metric("Rows", f"{eda['n_rows']:,}")
    m[1].metric("Columns", eda["n_columns"])
    m[2].metric("Target", eda["target"] or "—")
    m[3].metric("Positive rate", "—" if eda["positive_rate"] is None else f"{eda['positive_rate']:.1%}")
    m[4].metric("Duplicate rows", eda["duplicate_rows"])
    left, right = st.columns(2)
    with left:
        if eda["target_counts"]:
            counts = pd.DataFrame({"class": list(eda["target_counts"]), "count": list(eda["target_counts"].values())})
            st.plotly_chart(px.pie(counts, names="class", values="count", hole=0.5, title="Class balance"),
                            use_container_width=True)
    with right:
        if eda["missing_pct"]:
            miss = pd.Series(eda["missing_pct"]).sort_values(ascending=False).head(20)
            st.plotly_chart(px.bar(miss[::-1], orientation="h", title="Missing values (%)",
                                   labels={"value": "%", "index": "column"}), use_container_width=True)
        else:
            st.success("No missing values")
    corr = eda["correlation"]
    if corr["labels"]:
        fig = go.Figure(go.Heatmap(z=corr["matrix"], x=corr["labels"], y=corr["labels"], zmin=-1, zmax=1,
                                   colorscale="RdBu", reversescale=True))
        fig.update_layout(title="Correlation (top-variance numeric features + target)", height=600)
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Numeric summary"):
        st.dataframe(pd.DataFrame(eda["numeric_summary"]).T, use_container_width=True)
    return eda


def _render_column(client: ApiClient, dataset_id: str, target: str | None, eda: dict[str, Any]) -> None:
    names = [c["name"] for c in eda["columns"]]
    column = st.selectbox("Column", names, key="eda_column")
    try:
        prof = client.column_profile(dataset_id, column, target)
    except ApiError as exc:
        _show_error(exc)
        return
    if prof["kind"] == "numeric":
        edges = prof["edges"]
        mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
        fig = go.Figure()
        by_class = prof.get("counts_by_class")
        if by_class:
            for cls, counts in by_class.items():
                fig.add_bar(x=mids, y=counts, name=f"class {cls}")
            fig.update_layout(barmode="stack")
        else:
            fig.add_bar(x=mids, y=prof["counts"], name="count")
        fig.update_layout(title=f"Distribution of {column}", xaxis_title=column, yaxis_title="count")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Missing: {prof['missing']}")
        st.json(prof["stats"], expanded=False)
    else:
        fig = go.Figure(go.Bar(x=prof["categories"], y=prof["counts"], name="count"))
        if prof.get("churn_rate"):
            fig.add_scatter(x=prof["categories"], y=prof["churn_rate"], name="churn rate", yaxis="y2",
                            mode="lines+markers")
            fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "tickformat": ".0%", "rangemode": "tozero"})
        fig.update_layout(title=f"{column}: counts and churn rate", xaxis_title=column)
        st.plotly_chart(fig, use_container_width=True)


def _render_pca(client: ApiClient, dataset_id: str, target: str | None) -> None:
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Components", 2, 10, 3, key="pca_k")
    sample = c2.slider("Sample size", 500, 20_000, 3000, 500, key="pca_n")
    dim = c3.radio("Projection", ["2D", "3D"], horizontal=True, key="pca_dim")
    if not st.button("Compute PCA", key="pca_btn"):
        return
    try:
        with st.spinner("Computing PCA…"):
            res = client.pca(dataset_id, target, n_components=k, sample_size=sample)
    except ApiError as exc:
        _show_error(exc)
        return
    pts = pd.DataFrame(res["points"])
    pts["class"] = ["unlabelled" if lab is None else str(int(lab)) for lab in res["labels"]]
    evr = res["explained_variance_ratio"]
    axes = {f"PC{i + 1}": f"PC{i + 1} ({v:.1%})" for i, v in enumerate(evr)}
    pts = pts.rename(columns=axes)
    cols = list(axes.values())
    if dim == "3D" and len(cols) >= 3:
        fig = px.scatter_3d(pts, x=cols[0], y=cols[1], z=cols[2], color="class", opacity=0.6)
    else:
        fig = px.scatter(pts, x=cols[0], y=cols[1], color="class", opacity=0.6)
    fig.update_traces(marker={"size": 3})
    fig.update_layout(height=620, title=f"PCA of {res['n_points']:,} sampled rows")
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        ev = pd.DataFrame({"component": list(axes), "explained": evr})
        ev["cumulative"] = ev["explained"].cumsum()
        bar = px.bar(ev, x="component", y="explained", title="Explained variance")
        bar.add_scatter(x=ev["component"], y=ev["cumulative"], mode="lines+markers", name="cumulative")
        st.plotly_chart(bar, use_container_width=True)
    with right:
        for pc, rows in res["loadings"].items():
            load = pd.DataFrame(rows)
            st.plotly_chart(px.bar(load[::-1], x="loading", y="feature", orientation="h",
                                   title=f"Top loadings — {pc}"), use_container_width=True)


def render(client: ApiClient) -> None:
    """Render the EDA & PCA tab.

    Args:
        client: API client.
    """
    dataset_id = _dataset_picker(client)
    if not dataset_id:
        return
    target = st.text_input("Target column (blank = auto-detect)", key="eda_target").strip() or None
    sub = st.tabs(["Quality gates", "Overview", "Column explorer", "PCA"])
    with sub[0]:
        _render_quality(client, dataset_id, target)
    with sub[1]:
        eda = _render_summary(client, dataset_id, target)
        st.session_state["eda_cache"] = eda
    with sub[2]:
        cached = st.session_state.get("eda_cache")
        if cached:
            _render_column(client, dataset_id, target, cached)
    with sub[3]:
        _render_pca(client, dataset_id, target)
