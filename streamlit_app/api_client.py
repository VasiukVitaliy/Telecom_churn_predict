"""Pure HTTP client for the churn REST API.

This module (and the whole ``streamlit_app`` package) deliberately imports neither
scikit-learn, MLflow nor any ``ml_core`` module: the frontend talks to the backend
exclusively through REST.
"""

from __future__ import annotations

from typing import Any

import requests


class ApiError(RuntimeError):
    """Raised when the API is unreachable or answers with an error status.

    Attributes:
        status: HTTP status code (``0`` when the API could not be reached).
        payload: Decoded JSON body, if any.
    """

    def __init__(self, message: str, status: int = 0, payload: Any = None) -> None:
        super().__init__(message)
        self.status: int = status
        self.payload: Any = payload


class ApiClient:
    """Thin wrapper around :mod:`requests` with uniform error handling.

    Args:
        base_url: API root, e.g. ``http://localhost:8000``.
        timeout: Default request timeout in seconds.

    Example:
        >>> ApiClient("http://localhost:8000").base_url
        'http://localhost:8000'
    """

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.timeout: float = timeout
        self._session: requests.Session = requests.Session()

    # ------------------------------------------------------------------ core
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send a request and return the decoded JSON body.

        Args:
            method: HTTP verb.
            path: URL path beginning with ``/``.
            **kwargs: Forwarded to :meth:`requests.Session.request`.

        Returns:
            Decoded JSON (``dict`` / ``list``).

        Raises:
            ApiError: On connection problems or non-2xx responses.
        """
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = self._session.request(method, f"{self.base_url}{path}", **kwargs)
        except requests.RequestException as exc:
            raise ApiError(f"Cannot reach API at {self.base_url}: {exc}") from exc
        try:
            body: Any = response.json()
        except ValueError:
            body = None
        if not response.ok:
            detail = body.get("detail") if isinstance(body, dict) else None
            if isinstance(detail, list):  # FastAPI request-validation errors
                detail = "; ".join(f"{'.'.join(map(str, e.get('loc', [])))}: {e.get('msg')}" for e in detail)
            raise ApiError(str(detail or response.text or response.reason), response.status_code, body)
        return body

    def _get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def _post(self, path: str, json: Any = None, **params: Any) -> Any:
        return self._request("POST", path, json=json, params={k: v for k, v in params.items() if v is not None})

    # ---------------------------------------------------------------- system
    def health(self) -> dict[str, Any]:
        """Liveness and serving-model state."""
        return self._get("/health")

    # -------------------------------------------------------------- datasets
    def list_datasets(self) -> list[dict[str, Any]]:
        """List stored datasets."""
        return self._get("/datasets")

    def upload_dataset(self, filename: str, content: bytes) -> dict[str, Any]:
        """Upload a real dataset (CSV / Parquet)."""
        return self._request("POST", "/datasets/upload", files={"file": (filename, content)})

    def generate_synthetic(self, **body: Any) -> dict[str, Any]:
        """Generate a synthetic dataset (see ``SyntheticRequest``)."""
        return self._post("/datasets/synthetic", body)

    def validate_dataset(self, dataset_id: str, target_column: str | None = None) -> dict[str, Any]:
        """Infer the schema and run the quality gates."""
        return self._post("/datasets/validate", {"dataset_id": dataset_id, "target_column": target_column})

    def eda(self, dataset_id: str, target: str | None = None) -> dict[str, Any]:
        """Dataset-level EDA summary."""
        return self._get(f"/datasets/{dataset_id}/eda", target=target)

    def column_profile(self, dataset_id: str, column: str, target: str | None = None) -> dict[str, Any]:
        """Distribution of one column split by class."""
        return self._get(f"/datasets/{dataset_id}/columns/{column}", target=target)

    def pca(self, dataset_id: str, target: str | None = None, n_components: int = 3,
            sample_size: int = 3000) -> dict[str, Any]:
        """PCA projection of the preprocessed feature space."""
        return self._get(f"/datasets/{dataset_id}/pca", target=target, n_components=n_components,
                         sample_size=sample_size)

    # -------------------------------------------------------------- training
    def start_training(self, dataset_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Queue a background training job."""
        return self._post("/training/start", {"dataset_id": dataset_id, "config": config})

    def list_jobs(self) -> list[dict[str, Any]]:
        """All training jobs (newest first)."""
        return self._get("/training/jobs")

    def get_job(self, job_id: str) -> dict[str, Any]:
        """One training job."""
        return self._get(f"/training/jobs/{job_id}")

    # -------------------------------------------------------------- registry
    def model_versions(self) -> list[dict[str, Any]]:
        """Registered model versions with aliases and metrics."""
        return self._get("/models/versions")

    def model_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent MLflow runs."""
        return self._get("/models/runs", limit=limit)

    def set_alias(self, version: int, alias: str) -> dict[str, Any]:
        """Point ``alias`` at ``version``."""
        return self._post(f"/models/versions/{version}/alias", {"alias": alias})

    def promote(self, version: int) -> dict[str, Any]:
        """Set ``@champion`` to ``version``."""
        return self._post(f"/models/versions/{version}/promote")

    def delete_alias(self, alias: str) -> dict[str, Any]:
        """Delete an alias."""
        return self._request("DELETE", f"/models/aliases/{alias}")

    def model_info(self) -> dict[str, Any]:
        """Metadata of the serving model."""
        return self._get("/model/info")

    def model_schema(self) -> dict[str, Any]:
        """Input feature specification of the serving model."""
        return self._get("/model/schema")

    def reload(self, alias: str | None = None, version: int | None = None) -> dict[str, Any]:
        """Hot-reload the serving model."""
        return self._post("/model/reload", alias=alias, version=version)

    # ------------------------------------------------------------ prediction
    def predict(self, features: dict[str, Any], threshold: float | None = None) -> dict[str, Any]:
        """Score one subscriber."""
        return self._post("/predict", {"features": features, "threshold": threshold})

    def predict_batch(self, records: list[dict[str, Any]], threshold: float | None = None) -> dict[str, Any]:
        """Score many subscribers."""
        return self._post("/predict/batch", {"records": records, "threshold": threshold})

    def predict_upload(self, filename: str, content: bytes, threshold: float | None = None) -> dict[str, Any]:
        """Score an uploaded CSV / Parquet file."""
        return self._request("POST", "/predict/batch/upload", files={"file": (filename, content)},
                             params={"threshold": threshold} if threshold else None, timeout=300)

    def sensitivity(self, feature: str, features: dict[str, Any] | None = None, n_points: int = 25,
                    values: list[Any] | None = None, threshold: float | None = None) -> dict[str, Any]:
        """Probability curve for one feature."""
        return self._post("/predict/sensitivity", {
            "feature": feature, "features": features or {}, "n_points": n_points,
            "values": values, "threshold": threshold})

    # ------------------------------------------------------------ monitoring
    def monitoring_summary(self, all_versions: bool = False) -> dict[str, Any]:
        """Volumes, latency and live churn rate."""
        return self._get("/monitoring/summary", all_versions=str(all_versions).lower())

    def monitoring_timeseries(self, bucket_seconds: int = 60, n_buckets: int = 60) -> dict[str, Any]:
        """Bucketed telemetry."""
        return self._get("/monitoring/timeseries", bucket_seconds=bucket_seconds, n_buckets=n_buckets)

    def monitoring_drift(self, min_samples: int = 30) -> dict[str, Any]:
        """Feature drift and prediction shift."""
        return self._get("/monitoring/drift", min_samples=min_samples)

    def monitoring_recent(self, n: int = 200) -> list[dict[str, Any]]:
        """Most recent scored rows."""
        return self._get("/monitoring/recent", n=n)

    def monitoring_reset(self) -> dict[str, Any]:
        """Clear the in-memory telemetry window."""
        return self._post("/monitoring/reset")

    def simulate(self, n: int = 200, drift: bool = False, random_state: int | None = None) -> dict[str, Any]:
        """Generate demo traffic."""
        return self._post("/monitoring/simulate", {"n": n, "drift": drift, "random_state": random_state})
