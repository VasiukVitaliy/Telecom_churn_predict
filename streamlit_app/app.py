"""Multi-tab Streamlit dashboard (talks to the backend strictly through REST).

Run from ``app_ml_rest_api/``::

    streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make ``streamlit_app`` importable when launched as a script, independent of the CWD.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from streamlit_app import views_eda, views_monitoring, views_prediction, views_training  # noqa: E402
from streamlit_app.api_client import ApiClient, ApiError  # noqa: E402

DEFAULT_API_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")


def get_client() -> ApiClient:
    """Return the session-scoped API client (recreated when the URL changes).

    Returns:
        Configured :class:`ApiClient`.
    """
    url = st.session_state.get("api_url", DEFAULT_API_URL)
    client: ApiClient | None = st.session_state.get("client")
    if client is None or client.base_url != url.rstrip("/"):
        client = ApiClient(url)
        st.session_state["client"] = client
    return client


def sidebar(client: ApiClient) -> None:
    """Render connection settings and the serving-model status.

    Args:
        client: API client.
    """
    st.sidebar.title("📡 Telecom Churn")
    st.sidebar.text_input("API URL", key="api_url", value=DEFAULT_API_URL)
    try:
        health = client.health()
    except ApiError as exc:
        st.sidebar.error(f"API unreachable\n\n{exc}")
        return
    if health["status"] == "ok":
        st.sidebar.success("API healthy")
    else:
        st.sidebar.warning("API degraded (no model loaded or DB problem)")
    if health["model_loaded"]:
        st.sidebar.metric("Serving model", f"v{health['model_version']}", health.get("model_alias") or "by version")
    else:
        st.sidebar.info("No model serving. Train one, promote it to @champion and reload.")
    st.sidebar.caption(f"API v{health['api_version']} · uptime {health['uptime_s']:.0f}s")
    with st.sidebar.expander("Storage"):
        st.code(f"tracking: {health['tracking_uri']}\nartifacts: {health['artifact_root']}")


def main() -> None:
    """Dashboard entrypoint."""
    st.set_page_config(page_title="Telecom Churn ML", page_icon="📡", layout="wide")
    client = get_client()
    sidebar(client)
    st.title("Telecom Customer Churn — ML Console")
    tabs = st.tabs(["📊 EDA & PCA", "🧪 Training & Registry", "🎯 Prediction", "📈 Monitoring"])
    with tabs[0]:
        views_eda.render(client)
    with tabs[1]:
        views_training.render(client)
    with tabs[2]:
        views_prediction.render(client)
    with tabs[3]:
        views_monitoring.render(client)


if __name__ == "__main__":  # ``streamlit run`` executes the script as __main__
    main()
