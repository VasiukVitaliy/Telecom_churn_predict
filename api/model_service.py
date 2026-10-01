"""In-memory model manager with zero-downtime, alias-based hot reloading.

Concurrency model:
    * The active :class:`LoadedModel` is an *immutable snapshot*; readers grab the
      reference under a short lock and then predict without holding it.
    * A reload downloads + deserialises + dry-runs the candidate **outside** the
      read lock, then swaps the reference atomically. Requests already in flight
      finish on the old snapshot, new requests see the new one - nothing is dropped.
    * Concurrent reloads are serialised by a dedicated lock.
    * State lives on a service *instance* (owned by ``app.state``), not in module globals.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from ml_core.config import Settings, get_settings
from ml_core.data_validator import (
    ColumnKind, DataValidationError, DatasetSchema, normalize_frame, validate_dataframe,
)
from ml_core.registry import LoadedModel, ModelRegistry

logger = logging.getLogger(__name__)


class NoModelLoadedError(RuntimeError):
    """Raised when inference is requested before any model is loaded."""


class ModelValidationError(RuntimeError):
    """Raised when a candidate model fails the pre-swap dry-run."""


def feature_specs(model: LoadedModel) -> list[dict[str, Any]]:
    """Describe the model's input features (for forms and sensitivity analysis).

    Args:
        model: Loaded model snapshot.

    Returns:
        One dict per feature: ``name, kind, min, max, categories, default``.
    """
    schema = DatasetSchema.from_dict(model.schema)
    ref = model.reference_profile.get("features", {})
    specs: list[dict[str, Any]] = []
    for name in schema.feature_columns:
        col = schema.columns[name]
        if col.kind == ColumnKind.NUMERIC:
            mean = ref.get(name, {}).get("mean")
            if mean is None and col.min_value is not None and col.max_value is not None:
                mean = (col.min_value + col.max_value) / 2
            specs.append({"name": name, "kind": "numeric", "min": col.min_value, "max": col.max_value,
                          "categories": [], "default": mean})
        else:
            specs.append({"name": name, "kind": "categorical", "min": None, "max": None,
                          "categories": list(col.categories),
                          "default": col.categories[0] if col.categories else None})
    return specs


class ModelService:
    """Holds the serving model and performs atomic alias-based swaps.

    Args:
        registry: Registry facade used to fetch model versions.
        settings: Optional settings override.
    """

    def __init__(self, registry: ModelRegistry, settings: Settings | None = None) -> None:
        self._registry = registry
        self._settings = settings or get_settings()
        self._model: LoadedModel | None = None
        self._read_lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=50)

    # -------------------------------------------------------------- state access
    def is_loaded(self) -> bool:
        """Whether a model is currently serving."""
        with self._read_lock:
            return self._model is not None

    def current(self) -> LoadedModel:
        """Return the active model snapshot.

        Raises:
            NoModelLoadedError: If nothing has been loaded yet.
        """
        with self._read_lock:
            if self._model is None:
                raise NoModelLoadedError(
                    "No model is loaded. Train a model and promote it to @champion, then call POST /model/reload."
                )
            return self._model

    def history(self) -> list[dict[str, Any]]:
        """Recent reload events (newest last)."""
        return list(self._events)

    # ------------------------------------------------------------------- reload
    @staticmethod
    def _sample_frame(model: LoadedModel, n: int = 3) -> pd.DataFrame:
        specs = feature_specs(model)
        if not specs:
            raise ModelValidationError("Model has no schema metadata; cannot run a dry-run check.")
        return pd.DataFrame([{s["name"]: s["default"] for s in specs} for _ in range(n)])

    def _dry_run(self, model: LoadedModel) -> None:
        """Score a schema-derived sample and check output shape / range."""
        try:
            proba = model.pipeline.predict_proba(self._sample_frame(model))
        except ModelValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelValidationError(f"Dry-run prediction failed: {exc}") from exc
        arr = np.asarray(proba)
        if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise ModelValidationError(f"Dry-run produced invalid probabilities with shape {arr.shape}.")

    def load(self, alias: str | None = None, version: int | None = None) -> dict[str, Any]:
        """Load a registry model, validate it, and atomically swap it in.

        Args:
            alias: Alias to resolve (default: champion alias).
            version: Explicit version (overrides ``alias``).

        Returns:
            Event dictionary (new/previous version, alias, duration in ms).

        Raises:
            ModelNotFoundError: If the alias / version does not exist.
            RegistryError: If artifacts cannot be loaded.
            ModelValidationError: If the dry-run check fails (active model is untouched).
        """
        with self._reload_lock:
            started = time.perf_counter()
            candidate = self._registry.load_model(alias=alias, version=version)
            self._dry_run(candidate)
            with self._read_lock:
                previous, self._model = self._model, candidate
            event = {
                "at": time.time(), "model_version": candidate.version, "model_alias": candidate.alias,
                "previous_version": previous.version if previous else None,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            self._events.append(event)
            logger.info("Hot-reloaded model: %s", event)
            return event

    # ---------------------------------------------------------------- inference
    def prepare(
        self, records: list[dict[str, Any]] | pd.DataFrame, model: LoadedModel, *, single: bool
    ) -> tuple[pd.DataFrame, list[str]]:
        """Validate raw records against the model schema and return a model-ready frame.

        Single records may omit fields (they are imputed by the pipeline) as long as the
        missing share stays below ``MAX_MISSING_FRACTION``. Batches must contain every feature.

        Args:
            records: List of dicts or a dataframe.
            model: Model snapshot whose schema is enforced.
            single: ``True`` for one-record inference.

        Returns:
            ``(frame_with_feature_columns_only, warnings)``.

        Raises:
            DataValidationError: If any quality gate fails.
        """
        schema = DatasetSchema.from_dict(model.schema)
        features = schema.feature_columns
        frame = normalize_frame(records if isinstance(records, pd.DataFrame) else pd.DataFrame(records))
        warnings: list[str] = []
        if single:
            absent = [c for c in features if c not in frame.columns]
            for col in absent:
                frame[col] = np.nan
            blank = [c for c in features if frame[c].isna().all()]
            if blank and len(blank) / max(len(features), 1) > self._settings.max_missing_fraction:
                raise DataValidationError(
                    f"{len(blank)} of {len(features)} features are missing "
                    f"(limit {self._settings.max_missing_fraction:.0%})."
                )
            if blank:
                warnings.append(f"Imputed {len(blank)} missing feature(s): {blank[:8]}")
        report = validate_dataframe(
            frame, schema, mode="inference", settings=self._settings,
            check_missingness=not single, strict_extra_columns=single,
        )
        warnings.extend(report.warnings)
        return frame[features].reset_index(drop=True), warnings

    @staticmethod
    def score(model: LoadedModel, frame: pd.DataFrame) -> np.ndarray:
        """Return churn probabilities for ``frame`` using a specific model snapshot."""
        return np.asarray(model.pipeline.predict_proba(frame)[:, 1], dtype=float)
