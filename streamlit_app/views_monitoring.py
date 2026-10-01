"""Live telemetry, prediction shift and feature-drift dashboard (REST only)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.api_client import ApiClient, ApiError

STATUS_COLOR = {"stable": "#2ca02c", "warning": "#ff9900", "drift": "#d62728", "insufficient_data": "#9e9e9e"}
STATUS_ICON = {"stable": "🟢", "warning": "🟠", "drift": "🔴", "insufficient_data": "⚪"}


def _controls(client: ApiClient) -> None:
    with st.expander("🧪 Demo traffic & maintenance"):
        c1, c2, c3, c4 = st.columns(4)
        n = c1.number_input("Requests to simulate", 10, 5000, 300, step=50)
        drift = c2.checkbox("Inject drift", value=False)
        if c3.button("Send simulated traffic"):
            try:
                res = client.simulate(int(n), drift)
                st.success(f"Scored {res['simulated']} rows (positive rate {res['positive_rate']:.1%}).")
            except ApiError as exc:
                st.error(str(exc))
        if c4.button("Reset telemetry window"):
            try:
                client.monitoring_reset()
                st.success("Window cleared (audit log kept).")
            except ApiError as exc:
                st.error(str(exc))


def _summary(client: ApiClient) -> dict[str, Any] | None:
    try:
        s = client.monitoring_summary()
    except ApiError as exc:
        st.error(str(exc))
        return None
    m = st.columns(6)
    m[0].metric("Total predictions", f"{s['total_predictions']:,}")
    m[1].metric("Window rows", f"{s['window_predictions']:,}")
    m[2].metric("Requests / min", s["requests_last_minute"])
    live, exp = s["live_positive_rate"], s["expected_positive_rate"]
    m[3].metric("Live churn rate", "—" if live is None else f"{live:.1%}",
                None if s["rate_delta"] is None else f"{s['rate_delta'] * 100:+.1f} pp vs baseline",
                delta_color="inverse")
    m[4].metric("Expected churn rate", "—" if exp is None else f"{exp:.1%}")
    lat = s.get("latency_ms") or {}
    m[5].metric("Latency p95", f"{lat['p95']:.0f} ms" if lat else "—")
    return s


def _timeseries(client: ApiClient) -> None:
    c1, c2 = st.columns(2)
    bucket = c1.select_slider("Bucket width (s)", [5, 10, 30, 60, 300, 900, 3600], value=60, key="mon_bucket")
    buckets = c2.slider("Buckets", 5, 200, 60, key="mon_nb")
    try:
        ts = client.monitoring_timeseries(bucket, buckets)
    except ApiError as exc:
        st.error(str(exc))
        return
    data = pd.DataFrame(ts["buckets"])
    if data.empty:
        st.info("No traffic recorded yet — send a prediction or simulate traffic.")
        return
    data["t"] = pd.to_datetime(data["t"])
    fig = go.Figure()
    fig.add_bar(x=data["t"], y=data["predictions"], name="predictions", marker_color="#9ecae1", yaxis="y2")
    fig.add_scatter(x=data["t"], y=data["churn_rate"], name="bucket churn rate", mode="markers",
                    marker={"color": "#6baed6"})
    fig.add_scatter(x=data["t"], y=data["rolling_churn_rate"], name="rolling churn rate", mode="lines",
                    line={"color": "#08519c", "width": 3})
    if ts["baseline_rate"] is not None:
        fig.add_hline(y=ts["baseline_rate"], line_dash="dash", line_color="red",
                      annotation_text=f"baseline {ts['baseline_rate']:.1%}")
    fig.update_layout(
        title="Request volume and churn rate vs training baseline", height=420,
        yaxis={"title": "churn rate", "tickformat": ".0%", "rangemode": "tozero"},
        yaxis2={"title": "predictions", "overlaying": "y", "side": "right", "showgrid": False})
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(px.line(data, x="t", y="mean_probability", title="Mean predicted probability", markers=True),
                      use_container_width=True)
    right.plotly_chart(px.line(data, x="t", y="latency_p95_ms", title="Latency p95 (ms)", markers=True),
                       use_container_width=True)


def _drift(client: ApiClient) -> None:
    min_samples = st.slider("Minimum samples per feature", 10, 500, 30, key="drift_min")
    try:
        d = client.monitoring_drift(min_samples)
    except ApiError as exc:
        st.error(str(exc))
        return
    status = d["overall_status"]
    st.markdown(f"### {STATUS_ICON[status]} Overall status: **{status}** · window n = {d['n']}")
    if status == "insufficient_data":
        st.info(f"Drift needs at least {min_samples} scored rows for the serving model.")
        return
    shift = d.get("prediction_shift")
    if shift:
        c = st.columns(4)
        c[0].metric("Prediction PSI", "—" if shift["psi"] is None else f"{shift['psi']:.3f}")
        c[1].metric("KS p-value", "—" if shift["ks_pvalue"] is None else f"{shift['ks_pvalue']:.4f}")
        c[2].metric("Live positive rate", f"{shift['live_positive_rate']:.1%}")
        c[3].metric("Expected positive rate", f"{shift['expected_positive_rate']:.1%}")
    feats = pd.DataFrame([{"feature": k, **v} for k, v in d["features"].items()])
    if feats.empty:
        return
    feats = feats.sort_values("psi", ascending=False, na_position="last")
    fig = px.bar(feats.head(30)[::-1], x="psi", y="feature", orientation="h", color="status",
                 color_discrete_map=STATUS_COLOR, title="Feature drift — Population Stability Index")
    fig.add_vline(x=0.10, line_dash="dot", line_color="#ff9900")
    fig.add_vline(x=0.25, line_dash="dash", line_color="#d62728")
    fig.update_layout(height=max(350, 22 * min(len(feats), 30) + 120))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("PSI < 0.10 stable · 0.10–0.25 warning · ≥ 0.25 drift. KS p-value shown for numeric features.")
    st.dataframe(feats, use_container_width=True, hide_index=True)


@st.fragment(run_every=5)
def _live(client: ApiClient) -> None:
    """Auto-refreshing panel (summary + time series)."""
    _summary(client)
    _timeseries(client)


def render(client: ApiClient) -> None:
    """Render the Monitoring tab.

    Args:
        client: API client.
    """
    _controls(client)
    sub = st.tabs(["Live telemetry", "Drift analysis", "Recent predictions"])
    with sub[0]:
        if st.toggle("Auto-refresh every 5 s", value=True, key="mon_auto"):
            _live(client)
        else:
            _summary(client)
            _timeseries(client)
    with sub[1]:
        _drift(client)
    with sub[2]:
        try:
            recent = pd.DataFrame(client.monitoring_recent(300))
        except ApiError as exc:
            st.error(str(exc))
            return
        if recent.empty:
            st.info("No predictions yet.")
        else:
            recent["ts"] = pd.to_datetime(recent["ts"], unit="s")
            st.dataframe(recent.iloc[::-1], use_container_width=True, hide_index=True)
