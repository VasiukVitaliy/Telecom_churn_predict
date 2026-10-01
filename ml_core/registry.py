"""MLflow tracking + Model Registry integration with alias management.

Design notes:
    * Uses the explicit :class:`mlflow.tracking.MlflowClient` (no reliance on the
      global "active run"), so it is safe to call from background threads and
      from Optuna worker threads.
    * The relational backend (``MLFLOW_TRACKING_URI``) stores runs and the
      registry; ``MLFLOW_ARTIFACT_ROOT`` stores serialized pipelines and plots.
    * Serving reads models through aliases (``champion`` / ``challenger``);
      artifacts are downloaded into ``models/cache/`` and loaded from there.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import joblib
import mlflow.sklearn
import pandas as pd
from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from ml_core.config import Settings, get_settings

_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,255}$")
_RESERVED_ALIASES = {"latest"}
_METADATA_FILES: tuple[str, ...] = (
    "schema.json", "reference_profile.json", "training_summary.json", "feature_importance.json",
)


class RegistryError(RuntimeError):
    """Raised for registry / tracking failures (connection, missing alias, invalid input)."""


class ModelNotFoundError(RegistryError):
    """Raised when a requested alias or version does not exist."""


@dataclass
class LoadedModel:
    """A registry model version materialised in memory, with its metadata."""

    pipeline: Any
    name: str
    version: int
    run_id: str
    alias: str | None
    schema: dict[str, Any] = field(default_factory=dict)
    reference_profile: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    loaded_at: float = field(default_factory=time.time)

    @property
    def threshold(self) -> float:
        """Decision threshold stored at training time (default 0.5)."""
        return float(self.summary.get("threshold", 0.5))


class ModelRegistry:
    """Facade over MLflow tracking and the Model Registry.

    Args:
        settings: Configuration; defaults to :func:`~ml_core.config.get_settings`.

    Raises:
        RegistryError: If the database backend or artifact storage is unreachable.

    Example:
        >>> registry = ModelRegistry()  # doctest: +SKIP
        >>> registry.list_model_versions()  # doctest: +SKIP
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.settings.ensure_directories()
        self.model_name: str = self.settings.registered_model_name
        try:
            self.client: MlflowClient = MlflowClient(
                tracking_uri=self.settings.mlflow_tracking_uri,
                registry_uri=self.settings.mlflow_tracking_uri,
            )
            self.experiment_id: str = self._ensure_experiment()
        except MlflowException as exc:
            raise RegistryError(f"Cannot initialise MLflow backend: {exc}") from exc
        self.verify_connection()

    # ------------------------------------------------------------------ setup
    def _ensure_experiment(self) -> str:
        name = self.settings.mlflow_experiment_name
        experiment = self.client.get_experiment_by_name(name)
        if experiment is None:
            return self.client.create_experiment(name, artifact_location=self.settings.artifact_root_uri)
        if experiment.lifecycle_stage == "deleted":
            self.client.restore_experiment(experiment.experiment_id)
        return experiment.experiment_id

    def verify_connection(self) -> dict[str, Any]:
        """Check the DB backend and artifact directory are usable.

        Returns:
            Diagnostic dictionary (URIs, experiment id, writability).

        Raises:
            RegistryError: If the database query or artifact write test fails.
        """
        try:
            self.client.search_experiments(max_results=1)
        except Exception as exc:  # noqa: BLE001
            raise RegistryError(f"MLflow database unreachable ({self.settings.mlflow_tracking_uri}): {exc}") from exc
        root = self.settings.artifact_root_path
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            raise RegistryError(f"Artifact root not writable ({root}): {exc}") from exc
        return {
            "tracking_uri": self.settings.mlflow_tracking_uri, "artifact_root": str(root),
            "experiment_id": self.experiment_id, "registered_model": self.model_name, "ok": True,
        }

    # ---------------------------------------------------------------- tracking
    @contextmanager
    def run(self, run_name: str, tags: Mapping[str, str] | None = None) -> Iterator[str]:
        """Context manager creating a run and finalising its status.

        Args:
            run_name: Human-readable run name.
            tags: Extra run tags.

        Yields:
            The MLflow ``run_id``.
        """
        all_tags = {"mlflow.runName": run_name, **{k: str(v) for k, v in (tags or {}).items()}}
        run_id = self.client.create_run(self.experiment_id, tags=all_tags).info.run_id
        try:
            yield run_id
        except BaseException:
            self.client.set_terminated(run_id, "FAILED")
            raise
        self.client.set_terminated(run_id, "FINISHED")

    def log_params(self, run_id: str, params: Mapping[str, Any]) -> None:
        """Log parameters in batches (values truncated to MLflow's 500-char limit)."""
        items = [Param(str(k), str(v)[:500]) for k, v in params.items()]
        for i in range(0, len(items), 100):
            self.client.log_batch(run_id, params=items[i:i + 100])

    def log_metrics(self, run_id: str, metrics: Mapping[str, float], step: int = 0) -> None:
        """Log scalar metrics (non-finite values are skipped)."""
        ts = int(time.time() * 1000)
        items = [Metric(str(k), float(v), ts, step) for k, v in metrics.items()
                 if v is not None and float(v) == float(v) and abs(float(v)) != float("inf")]
        for i in range(0, len(items), 900):
            self.client.log_batch(run_id, metrics=items[i:i + 900])

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        """Set run tags."""
        self.client.log_batch(run_id, tags=[RunTag(str(k), str(v)[:5000]) for k, v in tags.items()])

    def log_trial(
        self, parent_run_id: str, trial_number: int, params: Mapping[str, Any],
        metrics: Mapping[str, float], duration_s: float, state: str = "COMPLETE",
    ) -> str:
        """Record one Optuna trial as a child run (thread-safe).

        Args:
            parent_run_id: Run id of the enclosing optimisation run.
            trial_number: Optuna trial number.
            params: Sampled hyper-parameters.
            metrics: Cross-validation metrics of the trial.
            duration_s: Trial wall-clock duration.
            state: Optuna trial state name.

        Returns:
            Child run id.
        """
        tags = {"mlflow.runName": f"trial-{trial_number:04d}", "mlflow.parentRunId": parent_run_id,
                "trial_state": state, "run_type": "optuna_trial"}
        child = self.client.create_run(self.experiment_id, tags=tags).info.run_id
        self.log_params(child, params)
        self.log_metrics(child, {**metrics, "duration_s": duration_s})
        self.client.set_terminated(child, "FINISHED" if state == "COMPLETE" else "FAILED")
        return child

    def log_figures(self, run_id: str, figures: Mapping[str, Any]) -> None:
        """Log matplotlib figures under ``plots/<name>.png``."""
        for name, fig in figures.items():
            self.client.log_figure(run_id, fig, f"plots/{name}.png")

    def log_json(self, run_id: str, name: str, payload: Any) -> None:
        """Log a JSON-serialisable object under ``metadata/<name>``."""
        self.client.log_dict(run_id, payload, f"metadata/{name}")

    def log_pipeline(self, run_id: str, pipeline: Any, X_example: pd.DataFrame) -> None:
        """Serialise the fitted pipeline (MLflow sklearn flavor + joblib) to the artifact store.

        Args:
            run_id: Target run.
            pipeline: Fitted (imblearn) pipeline.
            X_example: Raw feature sample used to infer the model signature.
        """
        sample = X_example.head(50)
        try:
            signature = infer_signature(sample, pipeline.predict(sample))
        except Exception:  # noqa: BLE001 - signature is optional metadata
            signature = None
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            mlflow.sklearn.save_model(pipeline, str(model_dir), signature=signature,
                                      serialization_format="cloudpickle")
            self.client.log_artifacts(run_id, str(model_dir), artifact_path="model")
            joblib.dump(pipeline, Path(tmp) / "pipeline.joblib", compress=3)
            self.client.log_artifact(run_id, str(Path(tmp) / "pipeline.joblib"), artifact_path="joblib")

    # ---------------------------------------------------------------- registry
    def register_version(self, run_id: str, tags: Mapping[str, str] | None = None,
                         auto_alias: str | None = None) -> int:
        """Register ``runs:/<run_id>/model`` as a new version of the registered model.

        Args:
            run_id: Run holding the ``model`` artifact.
            tags: Tags attached to the model version.
            auto_alias: Alias assigned immediately (default: challenger alias from settings).

        Returns:
            New model version number.

        Raises:
            RegistryError: On registry failure.
        """
        try:
            try:
                self.client.create_registered_model(self.model_name)
            except MlflowException:
                pass  # already exists
            mv = self.client.create_model_version(
                self.model_name, source=f"runs:/{run_id}/model", run_id=run_id,
                tags={k: str(v) for k, v in (tags or {}).items()},
            )
            for _ in range(60):
                if self.client.get_model_version(self.model_name, mv.version).status == "READY":
                    break
                time.sleep(0.5)
            version = int(mv.version)
            self.set_alias(auto_alias or self.settings.challenger_alias, version)
            return version
        except MlflowException as exc:
            raise RegistryError(f"Model registration failed: {exc}") from exc

    def set_alias(self, alias: str, version: int) -> None:
        """Point ``alias`` at ``version`` (atomic in the registry DB).

        Args:
            alias: Alias name (letters, digits, ``_`` or ``-``).
            version: Existing model version.

        Raises:
            RegistryError: If the alias is invalid.
            ModelNotFoundError: If the version does not exist.
        """
        if not _ALIAS_PATTERN.match(alias) or alias.lower() in _RESERVED_ALIASES or re.fullmatch(r"v\d+", alias):
            raise RegistryError(f"Invalid alias '{alias}'.")
        try:
            self.client.get_model_version(self.model_name, str(version))
        except MlflowException as exc:
            raise ModelNotFoundError(f"Model version {version} not found.") from exc
        self.client.set_registered_model_alias(self.model_name, alias, str(version))

    def delete_alias(self, alias: str) -> None:
        """Remove an alias (no-op semantics are not assumed; errors are wrapped)."""
        try:
            self.client.delete_registered_model_alias(self.model_name, alias)
        except MlflowException as exc:
            raise RegistryError(f"Could not delete alias '{alias}': {exc}") from exc

    def promote(self, version: int, alias: str | None = None) -> None:
        """Promote a version, by default to ``@champion``.

        Args:
            version: Model version to promote.
            alias: Alias override.
        """
        self.set_alias(alias or self.settings.champion_alias, version)

    def resolve_alias(self, alias: str) -> Any:
        """Return the model version object an alias points to.

        Raises:
            ModelNotFoundError: If the alias is unknown.
        """
        try:
            return self.client.get_model_version_by_alias(self.model_name, alias)
        except MlflowException as exc:
            raise ModelNotFoundError(f"Alias '@{alias}' not found for '{self.model_name}'.") from exc

    def list_model_versions(self) -> list[dict[str, Any]]:
        """List all registered versions (newest first) with aliases and metrics.

        Returns:
            List of dictionaries (version, run_id, aliases, metrics, params, tags, created_at).
        """
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            # ``search_model_versions`` does not populate aliases on the SQL store; the
            # registered-model record holds the authoritative ``{alias: version}`` mapping.
            alias_map = dict(self.client.get_registered_model(self.model_name).aliases or {})
        except MlflowException as exc:
            raise RegistryError(f"Cannot list model versions: {exc}") from exc
        rows: list[dict[str, Any]] = []
        for mv in sorted(versions, key=lambda v: int(v.version), reverse=True):
            try:
                data = self.client.get_run(mv.run_id).data
                metrics, params = dict(data.metrics), dict(data.params)
            except MlflowException:
                metrics, params = {}, {}
            rows.append({
                "version": int(mv.version), "run_id": mv.run_id, "status": mv.status,
                "aliases": sorted(a for a, v in alias_map.items() if int(v) == int(mv.version)), "tags": dict(mv.tags or {}),
                "metrics": metrics, "params": params, "created_at": int(mv.creation_timestamp) / 1000,
            })
        return rows

    def list_runs(self, max_results: int = 50) -> list[dict[str, Any]]:
        """List recent top-level training runs (Optuna trial child runs are excluded)."""
        runs = self.client.search_runs(
            [self.experiment_id], filter_string="tags.run_type = 'training'",
            order_by=["attributes.start_time DESC"], max_results=max_results,
        )
        return [{
            "run_id": r.info.run_id, "name": r.data.tags.get("mlflow.runName"), "status": r.info.status,
            "start_time": r.info.start_time / 1000, "metrics": dict(r.data.metrics),
            "params": dict(r.data.params),
        } for r in runs]

    # ----------------------------------------------------------------- loading
    def _read_metadata(self, run_id: str, dest: Path) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            local = Path(self.client.download_artifacts(run_id, "metadata", str(dest)))
        except Exception:  # noqa: BLE001
            return out
        for fname in _METADATA_FILES:
            fpath = local / fname
            if fpath.is_file():
                out[fname] = json.loads(fpath.read_text())
        return out

    def load_model(self, alias: str | None = None, version: int | None = None) -> LoadedModel:
        """Download and deserialise a model version (by alias or explicit version).

        Args:
            alias: Alias to resolve (defaults to the champion alias when ``version`` is None).
            version: Explicit version number (takes precedence over ``alias``).

        Returns:
            :class:`LoadedModel` with pipeline, schema, reference profile and summary.

        Raises:
            ModelNotFoundError: If the alias / version does not exist.
            RegistryError: If artifacts cannot be downloaded or deserialised.
        """
        if version is None:
            alias = alias or self.settings.champion_alias
            mv = self.resolve_alias(alias)
        else:
            try:
                mv = self.client.get_model_version(self.model_name, str(version))
            except MlflowException as exc:
                raise ModelNotFoundError(f"Model version {version} not found.") from exc
        dest = self.settings.model_cache_dir / f"{self.model_name}" / f"v{mv.version}"
        dest.mkdir(parents=True, exist_ok=True)
        try:
            model_path = Path(self.client.download_artifacts(mv.run_id, "model", str(dest)))
            pipeline = mlflow.sklearn.load_model(str(model_path))
        except Exception as exc:  # noqa: BLE001
            raise RegistryError(f"Failed to load model version {mv.version}: {exc}") from exc
        meta = self._read_metadata(mv.run_id, dest)
        return LoadedModel(
            pipeline=pipeline, name=self.model_name, version=int(mv.version), run_id=mv.run_id,
            alias=alias if version is None else None,
            schema=meta.get("schema.json", {}), reference_profile=meta.get("reference_profile.json", {}),
            summary=meta.get("training_summary.json", {}),
            feature_importance=meta.get("feature_importance.json", []),
        )
