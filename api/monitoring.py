"""Inference telemetry: request logging, rolling metrics, prediction shift and feature drift.

Every scored request is kept in a bounded in-memory window (for live dashboards)
and appended to a JSONL audit log so telemetry survives restarts.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from api.schemas import sanitize
from ml_core.config import Settings, get_settings
from ml_core.data_validator import compute_feature_drift, compute_prediction_shift

logger = logging.getLogger(__name__)
_SEVERITY = {"insufficient_data": 0, "stable": 1, "warning": 2, "drift": 3}
_MAX_LOG_BYTES = 50 * 1024 * 1024


class InferenceMonitor:
    """Thread-safe telemetry collector.

    Args:
        settings: Settings supplying window size and PSI thresholds.
        log_path: JSONL log location (default ``models/monitoring/inference_log.jsonl``).
    """

    def __init__(self, settings: Settings | None = None, log_path: Path | None = None) -> None:
        self._settings = settings or get_settings()
        self._records: deque[dict[str, Any]] = deque(maxlen=self._settings.monitor_window_size)
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_predictions = 0
        self._log_path = log_path or (self._settings.models_dir / "monitoring" / "inference_log.jsonl")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- recording
    def record(
        self, *, request_id: str, model_version: int, latency_ms: float,
        features: Sequence[dict[str, Any]], probabilities: Sequence[float], labels: Sequence[int],
        ts: float | None = None, persist: bool = True,
    ) -> None:
        """Log one inference request (one or many rows).

        Args:
            request_id: Unique request identifier.
            model_version: Version that produced the scores.
            latency_ms: End-to-end scoring latency of the request.
            features: Raw feature dicts, one per scored row.
            probabilities: Predicted churn probabilities.
            labels: Thresholded predictions.
            ts: Unix timestamp (defaults to now).
            persist: Append to the JSONL audit log.
        """
        stamp = ts or time.time()
        rows = [{
            "ts": stamp, "request_id": request_id, "model_version": model_version, "latency_ms": latency_ms,
            "proba": float(p), "label": int(l), "features": sanitize(f),
        } for f, p, l in zip(features, probabilities, labels)]
        with self._lock:
            self._records.extend(rows)
            self._total_requests += 1
            self._total_predictions += len(rows)
        if persist:
            self._append_log({"ts": stamp, "request_id": request_id, "model_version": model_version,
                              "latency_ms": latency_ms, "probabilities": [float(p) for p in probabilities],
                              "labels": [int(x) for x in labels], "features": sanitize(list(features))})

    def _append_log(self, entry: dict[str, Any]) -> None:
        try:
            with self._lock:
                if self._log_path.exists() and self._log_path.stat().st_size > _MAX_LOG_BYTES:
                    self._log_path.replace(self._log_path.with_suffix(".jsonl.1"))
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        except (OSError, ValueError) as exc:
            logger.warning("Could not append to inference log: %s", exc)

    def load_history(self) -> int:
        """Restore the in-memory window from the tail of the JSONL log.

        Returns:
            Number of prediction rows restored.
        """
        if not self._log_path.is_file():
            return 0
        lines: deque[str] = deque(maxlen=max(self._settings.monitor_window_size, 1))
        with self._log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                lines.append(line)
        restored = 0
        for line in lines:
            try:
                e = json.loads(line)
                self.record(request_id=e["request_id"], model_version=e["model_version"],
                            latency_ms=e["latency_ms"], features=e["features"],
                            probabilities=e["probabilities"], labels=e["labels"], ts=e["ts"], persist=False)
                restored += len(e["probabilities"])
            except (KeyError, ValueError, TypeError):
                continue
        return restored

    def reset(self) -> None:
        """Clear the in-memory window and counters (the audit log is kept)."""
        with self._lock:
            self._records.clear()
            self._total_requests = self._total_predictions = 0

    def _snapshot(self, model_version: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._records)
        return [r for r in rows if model_version is None or r["model_version"] == model_version]

    # ----------------------------------------------------------------- analytics
    def summary(self, baseline_rate: float | None, model_version: int | None = None) -> dict[str, Any]:
        """Aggregate window statistics.

        Args:
            baseline_rate: Expected positive rate from the training reference.
            model_version: Restrict to one version (``None`` = all).

        Returns:
            Volumes, latency percentiles, live vs expected churn rate.
        """
        rows = self._snapshot(model_version)
        now = time.time()
        with self._lock:
            totals = {"total_requests": self._total_requests, "total_predictions": self._total_predictions}
        if not rows:
            return {**totals, "window_predictions": 0, "window_requests": 0, "requests_last_minute": 0,
                    "live_positive_rate": None, "expected_positive_rate": baseline_rate,
                    "rate_delta": None, "mean_probability": None, "latency_ms": {}}
        per_request = {r["request_id"]: (r["ts"], r["latency_ms"]) for r in rows}
        lat = np.array([v[1] for v in per_request.values()])
        live_rate = float(np.mean([r["label"] for r in rows]))
        return {
            **totals, "window_predictions": len(rows), "window_requests": len(per_request),
            "requests_last_minute": int(sum(1 for t, _ in per_request.values() if t >= now - 60)),
            "live_positive_rate": live_rate, "expected_positive_rate": baseline_rate,
            "rate_delta": None if baseline_rate is None else live_rate - baseline_rate,
            "mean_probability": float(np.mean([r["proba"] for r in rows])),
            "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                           "p99": float(np.percentile(lat, 99)), "mean": float(lat.mean())},
            "by_model_version": pd.Series([r["model_version"] for r in rows]).value_counts().to_dict(),
        }

    def timeseries(self, bucket_seconds: int = 60, n_buckets: int = 60, rolling: int = 5) -> list[dict[str, Any]]:
        """Time-bucketed request volume, churn rate (bucket + rolling) and latency.

        Args:
            bucket_seconds: Bucket width.
            n_buckets: Number of most recent non-empty buckets to return.
            rolling: Number of buckets in the rolling churn-rate window.

        Returns:
            Chronological list of bucket dictionaries.
        """
        rows = self._snapshot()
        if not rows:
            return []
        frame = pd.DataFrame(rows)
        frame["bucket"] = (frame["ts"] // bucket_seconds).astype(int)
        grouped = frame.groupby("bucket")
        out = pd.DataFrame({
            "requests": grouped["request_id"].nunique(), "predictions": grouped.size(),
            "positives": grouped["label"].sum(), "mean_probability": grouped["proba"].mean(),
            "latency_p95_ms": grouped["latency_ms"].quantile(0.95),
        }).sort_index().tail(n_buckets)
        out["churn_rate"] = out["positives"] / out["predictions"]
        out["rolling_churn_rate"] = (
            out["positives"].rolling(rolling, min_periods=1).sum()
            / out["predictions"].rolling(rolling, min_periods=1).sum()
        )
        out["t"] = [datetime.fromtimestamp(b * bucket_seconds, tz=timezone.utc).isoformat() for b in out.index]
        return sanitize(out.reset_index(drop=True).to_dict("records"))

    def drift(self, reference: dict[str, Any], model_version: int | None = None, min_samples: int = 30) -> dict[str, Any]:
        """Feature drift (PSI + KS) and prediction shift versus the training reference.

        Args:
            reference: Reference profile stored with the model.
            model_version: Restrict live data to one model version.
            min_samples: Minimum rows required before drift is evaluated.

        Returns:
            ``{"n", "overall_status", "features", "prediction_shift"}``.
        """
        rows = self._snapshot(model_version)
        if not reference or len(rows) < min_samples:
            return {"n": len(rows), "overall_status": "insufficient_data", "features": {}, "prediction_shift": None}
        live = pd.DataFrame([r["features"] for r in rows])
        feats = compute_feature_drift(
            reference, live, min_samples=min_samples,
            psi_warning=self._settings.psi_warning, psi_alert=self._settings.psi_alert,
        )
        shift = None
        if "predictions" in reference:
            shift = compute_prediction_shift(reference["predictions"], np.array([r["proba"] for r in rows]))
        worst = max((v["status"] for v in feats.values()), key=lambda s: _SEVERITY[s], default="insufficient_data")
        if shift and shift["psi"] is not None:
            p = shift["psi"]
            level = "drift" if p >= self._settings.psi_alert else "warning" if p >= self._settings.psi_warning else "stable"
            worst = max([worst, level], key=lambda s: _SEVERITY[s])
        return {"n": len(rows), "overall_status": worst, "features": feats, "prediction_shift": shift}

    def recent(self, n: int = 100) -> list[dict[str, Any]]:
        """Most recent scored rows (without feature payloads)."""
        rows = self._snapshot()[-n:]
        return [{k: v for k, v in r.items() if k != "features"} for r in rows]
