"""Single / batch inference and sensitivity analysis (REST only)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.api_client import ApiClient, ApiError


def _error(exc: ApiError) -> None:
    st.error(str(exc))
    if exc.status == 503:
        st.info("Promote a model to @champion and reload it from the Training & Registry tab.")


def _threshold_slider(default: float, key: str) -> float:
    return st.slider("Decision threshold", 0.05, 0.95, float(default), 0.01, key=key)


def _record_form(specs: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Dynamic input widgets for every model feature.

    Args:
        specs: ``/model/schema`` feature specs.

    Returns:
        ``(record, omitted_features)``. Omitted features are left to the server-side imputer.
    """
    omit = st.multiselect("Leave these features empty (imputed by the pipeline)",
                          [s["name"] for s in specs], key="pred_omit")
    record: dict[str, Any] = {}
    cols = st.columns(3)
    for i, spec in enumerate(specs):
        name = spec["name"]
        with cols[i % 3]:
            if name in omit:
                st.text_input(name, value="", disabled=True, key=f"f_{name}_off")
                continue
            if spec["kind"] == "numeric":
                lo, hi, default = spec["min"], spec["max"], spec["default"]
                default = 0.0 if default is None else float(default)
                if lo is not None and hi is not None and hi > lo:
                    record[name] = st.number_input(name, value=float(min(max(default, lo), hi)), min_value=float(lo),
                                                   max_value=float(hi), key=f"f_{name}")
                else:
                    record[name] = st.number_input(name, value=default, key=f"f_{name}")
            else:
                cats = spec["categories"] or [str(spec["default"])]
                idx = cats.index(spec["default"]) if spec["default"] in cats else 0
                record[name] = st.selectbox(name, cats, index=idx, key=f"f_{name}")
    return record, omit


def _single(client: ApiClient, specs: list[dict[str, Any]], model_threshold: float) -> None:
    record, _ = _record_form(specs)
    st.session_state["base_record"] = record
    threshold = _threshold_slider(model_threshold, "pred_thr")
    if not st.button("Predict churn", type="primary", key="predict_btn"):
        return
    try:
        res = client.predict(record, threshold)
    except ApiError as exc:
        _error(exc)
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=res["churn_probability"] * 100, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "threshold": {"line": {"color": "red", "width": 3},
                                                                "value": res["threshold"] * 100},
                   "bar": {"color": "crimson" if res["prediction"] else "seagreen"}},
            title={"text": "Churn probability"}))
        gauge.update_layout(height=300)
        st.plotly_chart(gauge, use_container_width=True)
    with c2:
        (st.error if res["prediction"] else st.success)(
            f"Prediction: {'WILL CHURN' if res['prediction'] else 'will stay'} "
            f"(p={res['churn_probability']:.3f}, threshold={res['threshold']:.2f})")
        st.caption(f"model v{res['model_version']} @{res['model_alias']} · {res['latency_ms']} ms · "
                   f"request {res['request_id']}")
        for warning in res["warnings"]:
            st.warning(warning)


def _batch(client: ApiClient, model_threshold: float) -> None:
    st.caption("Upload a CSV / Parquet file containing every model feature column.")
    upload = st.file_uploader("Batch file", type=["csv", "tsv", "parquet"], key="batch_upload")
    threshold = _threshold_slider(model_threshold, "batch_thr")
    if upload is None or not st.button("Score file", type="primary", key="batch_btn"):
        return
    try:
        with st.spinner("Scoring…"):
            res = client.predict_upload(upload.name, upload.getvalue(), threshold)
    except ApiError as exc:
        _error(exc)
        return
    m = st.columns(4)
    m[0].metric("Rows scored", f"{res['n']:,}")
    m[1].metric("Predicted churn rate", f"{res['positive_rate']:.1%}")
    m[2].metric("Latency", f"{res['latency_ms']:.0f} ms")
    m[3].metric("Model", f"v{res['model_version']}")
    for warning in res["warnings"]:
        st.warning(warning)
    frame = pd.DataFrame(res["predictions"])
    st.plotly_chart(px.histogram(frame, x="churn_probability", nbins=40, title="Probability distribution")
                    .add_vline(x=res["threshold"], line_color="red"), use_container_width=True)
    st.dataframe(frame.sort_values("churn_probability", ascending=False).head(500),
                 use_container_width=True, hide_index=True)
    st.download_button("⬇ Download predictions (CSV)", frame.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")


def _sensitivity(client: ApiClient, specs: list[dict[str, Any]], model_threshold: float) -> None:
    base = st.session_state.get("base_record", {})
    if not base:
        st.info("Open the *Single prediction* tab once to define a base subscriber "
                "(otherwise model defaults are used).")
    c1, c2, c3 = st.columns(3)
    feature = c1.selectbox("Feature to vary", [s["name"] for s in specs], key="sens_feature")
    points = c2.slider("Grid points (numeric)", 3, 100, 25, key="sens_points")
    threshold = c3.slider("Threshold", 0.05, 0.95, float(model_threshold), 0.01, key="sens_thr")
    if not st.button("Compute sensitivity curve", key="sens_btn"):
        return
    try:
        res = client.sensitivity(feature, {k: v for k, v in base.items() if k != feature}, points, None, threshold)
    except ApiError as exc:
        _error(exc)
        return
    if res["kind"] == "numeric":
        fig = px.line(x=res["values"], y=res["probabilities"], markers=True,
                      labels={"x": feature, "y": "churn probability"},
                      title=f"Churn probability vs {feature}")
    else:
        fig = px.bar(x=[str(v) for v in res["values"]], y=res["probabilities"],
                     labels={"x": feature, "y": "churn probability"}, title=f"Churn probability by {feature}")
    fig.add_hline(y=res["threshold"], line_dash="dash", line_color="red", annotation_text="threshold")
    fig.add_hline(y=res["base_probability"], line_dash="dot", line_color="gray", annotation_text="base record")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"model v{res['model_version']} · base probability {res['base_probability']:.3f}")


def render(client: ApiClient) -> None:
    """Render the Prediction tab.

    Args:
        client: API client.
    """
    try:
        schema = client.model_schema()
    except ApiError as exc:
        _error(exc)
        return
    specs, thr = schema["features"], schema["threshold"]
    st.caption(f"Serving model v{schema['version']} · {len(specs)} input features")
    sub = st.tabs(["Single prediction", "Batch prediction", "Sensitivity analysis"])
    with sub[0]:
        _single(client, specs, thr)
    with sub[1]:
        _batch(client, thr)
    with sub[2]:
        _sensitivity(client, specs, thr)
