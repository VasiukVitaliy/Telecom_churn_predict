"""Pydantic v2 request / response contracts and NaN-safe JSON serialisation."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ml_core.trainer import TrainingConfig


def sanitize(obj: Any) -> Any:
    """Recursively convert numpy / pandas / non-finite values into strict-JSON types.

    Args:
        obj: Arbitrary nested structure.

    Returns:
        Structure containing only JSON-safe values (``NaN``/``inf`` become ``None``).
    """
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


class SafeJSONResponse(JSONResponse):
    """JSON response that never fails on ``NaN`` / ``inf`` (they are emitted as ``null``)."""

    def render(self, content: Any) -> bytes:
        """Serialise ``content`` with strict JSON rules after sanitising."""
        return json.dumps(sanitize(content), ensure_ascii=False, allow_nan=False).encode("utf-8")


class RequestModel(BaseModel):
    """Base for request bodies: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ResponseModel(BaseModel):
    """Base for response bodies."""

    model_config = ConfigDict(protected_namespaces=())


# ----------------------------------------------------------------------------- health
class HealthResponse(ResponseModel):
    """Service liveness and model state."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: int | None = None
    model_alias: str | None = None
    tracking_uri: str
    artifact_root: str
    uptime_s: float
    api_version: str


# --------------------------------------------------------------------------- datasets
class DatasetInfo(ResponseModel):
    """Stored dataset descriptor."""

    dataset_id: str
    source: Literal["raw", "synthetic"]
    filename: str
    size_bytes: int
    n_rows: int | None = None
    n_columns: int | None = None


class SyntheticRequest(RequestModel):
    """Parameters of the synthetic subscriber generator."""

    n_samples: int = Field(20_000, ge=500, le=200_000)
    target_churn_rate: float = Field(0.20, ge=0.05, le=0.5)
    missing_rate: float = Field(0.0, ge=0.0, le=0.5)
    random_state: int = 42
    filename: str | None = Field(None, pattern=r"^[A-Za-z0-9_\-]+\.(csv|parquet)$")


class ValidateRequest(RequestModel):
    """Dataset validation request."""

    dataset_id: str
    target_column: str | None = None


class TrainRequest(RequestModel):
    """Start a background training job."""

    dataset_id: str
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class TrainStartResponse(ResponseModel):
    """Acknowledgement of a queued training job."""

    job_id: str
    status: str


# ----------------------------------------------------------------------------- models
class AliasRequest(RequestModel):
    """Assign an alias (e.g. ``champion``) to a model version."""

    alias: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")


class ReloadResponse(ResponseModel):
    """Result of a hot reload."""

    reloaded: bool
    model_version: int
    model_alias: str | None
    previous_version: int | None
    duration_ms: float


# -------------------------------------------------------------------------- prediction
class PredictRequest(RequestModel):
    """Single-subscriber inference request (missing fields are imputed)."""

    features: dict[str, Any] = Field(min_length=1)
    threshold: float | None = Field(None, gt=0.0, lt=1.0)


class PredictResponse(ResponseModel):
    """Single-subscriber inference result."""

    request_id: str
    churn_probability: float
    prediction: int
    threshold: float
    model_version: int
    model_alias: str | None
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)


class BatchPredictRequest(RequestModel):
    """Batch inference request."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=50_000)
    threshold: float | None = Field(None, gt=0.0, lt=1.0)


class BatchItem(ResponseModel):
    """One scored row of a batch."""

    index: int
    churn_probability: float
    prediction: int


class BatchPredictResponse(ResponseModel):
    """Batch inference result."""

    request_id: str
    n: int
    positive_rate: float
    threshold: float
    model_version: int
    model_alias: str | None
    latency_ms: float
    predictions: list[BatchItem]
    warnings: list[str] = Field(default_factory=list)


class SensitivityRequest(RequestModel):
    """Vary one feature of a base record and observe the churn probability."""

    features: dict[str, Any] = Field(default_factory=dict)
    feature: str
    values: list[Any] | None = None
    n_points: int = Field(25, ge=3, le=200)
    threshold: float | None = Field(None, gt=0.0, lt=1.0)


class SensitivityResponse(ResponseModel):
    """Probability curve for one feature."""

    feature: str
    kind: Literal["numeric", "categorical"]
    values: list[Any]
    probabilities: list[float]
    base_probability: float
    threshold: float
    model_version: int


class SimulateRequest(RequestModel):
    """Generate demo traffic (requires a model trained on the synthetic schema)."""

    n: int = Field(200, ge=10, le=5000)
    drift: bool = False
    random_state: int | None = None
