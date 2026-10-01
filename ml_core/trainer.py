"""Training engine: baseline and Optuna modes, CV, evaluation, MLflow logging, job manager.

Leakage guarantees:
    * The stratified train/test split happens first; the test set is used exactly once.
    * Imputation, encoding, scaling and oversampling all live in one
      ``imblearn.pipeline.Pipeline`` that is re-fitted on each CV training fold.
    * Optuna optimises CV performance on the *training* split only.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

import numpy as np
import optuna
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml_core.config import Settings, get_settings
from ml_core.data_generator import SamplingStrategy, build_sampler, uses_class_weight
from ml_core.data_validator import (
    DatasetSchema, build_prediction_profile, build_reference_profile, prepare_training_frame,
)
from ml_core.models import SUPPORTED_MODELS, ModelName, build_classifier, suggest_params
from ml_core.preprocessor import (
    CategoricalImputation, NumericImputation, build_preprocessor, get_feature_names,
)
from ml_core.registry import ModelRegistry

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ProgressCallback = Callable[[dict[str, Any]], None]
OptimizeMetric = Literal["roc_auc", "pr_auc", "f1"]


class TrainingError(RuntimeError):
    """Raised when training or optimisation cannot complete."""


class TrainingConfig(BaseModel):
    """User-facing training configuration (validated; also the REST payload).

    Attributes:
        mode: ``baseline`` (fixed defaults) or ``optuna`` (hyper-parameter search).
        param_space: Optional per-parameter Optuna spec overrides.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: ModelName = "lightgbm"
    mode: Literal["baseline", "optuna"] = "baseline"
    target_column: str | None = None
    test_size: float = Field(0.2, gt=0.0, lt=0.5)
    cv_folds: int = Field(5, ge=2, le=20)
    sampling: SamplingStrategy = "class_weight"
    sampling_ratio: float = Field(0.6, gt=0.0, le=1.0)
    numeric_imputation: NumericImputation = "median"
    categorical_imputation: CategoricalImputation = "most_frequent"
    scale_numeric: bool = False
    decision_threshold: float = Field(0.5, gt=0.0, lt=1.0)
    n_trials: int = Field(25, ge=1, le=500)
    timeout_seconds: int | None = Field(None, ge=10)
    optimize_metric: OptimizeMetric = "pr_auc"
    optuna_n_jobs: int = Field(1, ge=1, le=16)
    param_space: dict[str, dict[str, Any]] | None = None
    run_name: str | None = None
    register: bool = True
    random_state: int = 42

    @field_validator("model_name")
    @classmethod
    def _known_model(cls, value: str) -> str:
        """Reject unsupported algorithms early."""
        if value not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{value}'.")
        return value

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> "TrainingConfig":
        """Create a config seeded with the environment defaults.

        Args:
            settings: Application settings.
            **overrides: Field overrides.

        Returns:
            Validated :class:`TrainingConfig`.
        """
        base: dict[str, Any] = {
            "test_size": settings.test_size, "cv_folds": settings.cv_folds,
            "sampling": settings.sampling_strategy, "sampling_ratio": settings.sampling_ratio,
            "numeric_imputation": settings.numeric_imputation,
            "categorical_imputation": settings.categorical_imputation,
            "optuna_n_jobs": settings.optuna_n_jobs, "random_state": settings.random_state,
        }
        return cls(**{**base, **overrides})


@dataclass
class TrainingResult:
    """Outcome of one training execution."""

    run_id: str
    model_name: str
    mode: str
    model_version: int | None
    test_metrics: dict[str, float]
    cv_metrics: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    best_params: dict[str, Any]
    n_trials: int
    threshold: float
    duration_s: float
    n_train: int
    n_test: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# Metrics & CV
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray | pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute the full binary-classification metric set.

    Args:
        y_true: Ground-truth labels (0/1).
        proba: Predicted probability of class 1.
        threshold: Decision threshold for label-based metrics.

    Returns:
        Dict with ``roc_auc, pr_auc, f1, precision, recall, accuracy, brier_score``.

    Raises:
        ValueError: If only one class is present in ``y_true``.
    """
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Metrics need both classes in y_true.")
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier_score": float(brier_score_loss(y, proba)),
    }


def cross_validate_pipeline(
    pipeline: ImbPipeline, X: pd.DataFrame, y: pd.Series, *, n_splits: int, random_state: int,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Stratified K-fold CV; the whole pipeline is re-fitted on every training fold.

    Args:
        pipeline: Unfitted pipeline (cloned per fold).
        X: Training features.
        y: Training labels.
        n_splits: Number of folds.
        random_state: Shuffle seed.
        threshold: Decision threshold for label-based metrics.

    Returns:
        ``{metric: {"mean": float, "std": float}}``.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict[str, float]] = []
    for tr_idx, va_idx in skf.split(X, y):
        fold = clone(pipeline).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        proba = fold.predict_proba(X.iloc[va_idx])[:, 1]
        rows.append(compute_metrics(y.iloc[va_idx], proba, threshold))
    frame = pd.DataFrame(rows)
    return {c: {"mean": float(frame[c].mean()), "std": float(frame[c].std(ddof=0))} for c in frame.columns}


def build_pipeline(
    schema: DatasetSchema, config: TrainingConfig, params: dict[str, Any] | None, *,
    minority_ratio: float, scale_pos_weight: float, settings: Settings, n_jobs: int,
) -> ImbPipeline:
    """Assemble ``preprocess -> [sampler] -> classifier`` as an imblearn pipeline.

    Args:
        schema: Inferred dataset schema.
        config: Training configuration.
        params: Classifier hyper-parameters (``None`` = baseline defaults).
        minority_ratio: Current minority/majority ratio of the training split.
        scale_pos_weight: Negative/positive ratio (XGBoost cost-sensitive learning).
        settings: Application settings.
        n_jobs: Threads per estimator.

    Returns:
        Unfitted pipeline. Oversampling is skipped automatically at predict time.
    """
    steps: list[tuple[str, Any]] = [(
        "preprocess",
        build_preprocessor(
            schema, numeric_imputation=config.numeric_imputation,
            categorical_imputation=config.categorical_imputation, scale_numeric=config.scale_numeric,
            max_categories=settings.max_categorical_cardinality, random_state=config.random_state,
        ),
    )]
    sampler = build_sampler(
        config.sampling, sampling_ratio=config.sampling_ratio, random_state=config.random_state,
        minority_ratio=minority_ratio,
    )
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", build_classifier(
        config.model_name, params, random_state=config.random_state,
        class_weight_balanced=uses_class_weight(config.sampling),
        scale_pos_weight=scale_pos_weight, n_jobs=n_jobs,
    )))
    return ImbPipeline(steps)


def extract_feature_importance(pipeline: ImbPipeline, top_n: int = 30) -> list[dict[str, Any]]:
    """Return normalised feature importances of a fitted pipeline (empty if unsupported)."""
    clf = pipeline.named_steps["clf"]
    raw = getattr(clf, "feature_importances_", None)
    if raw is None:
        return []
    names = get_feature_names(pipeline.named_steps["preprocess"])
    values = np.asarray(raw, dtype=float)
    values = values / values.sum() if values.sum() > 0 else values
    order = np.argsort(values)[::-1][:top_n]
    return [{"feature": names[i], "importance": float(values[i])} for i in order]


def _build_figures(y_true: np.ndarray, proba: np.ndarray, cm: list[list[int]]) -> dict[str, Figure]:
    """Create confusion-matrix and ROC / PR figures (pyplot-free, thread-safe)."""
    cm_fig = Figure(figsize=(4.2, 3.8))
    ax = cm_fig.subplots()
    ax.imshow(np.asarray(cm), cmap="Blues")
    ax.set_xticks([0, 1], ["Stay", "Churn"])
    ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    cm_fig.tight_layout()

    curves = Figure(figsize=(8, 3.6))
    ax_roc, ax_pr = curves.subplots(1, 2)
    fpr, tpr, _ = roc_curve(y_true, proba)
    ax_roc.plot(fpr, tpr)
    ax_roc.plot([0, 1], [0, 1], "--", color="grey")
    ax_roc.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ax_pr.plot(rec, prec)
    ax_pr.set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    curves.tight_layout()
    return {"confusion_matrix": cm_fig, "roc_pr_curves": curves}


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class ModelTrainer:
    """Runs baseline or Optuna training and records everything in MLflow.

    Args:
        registry: Tracking / registry facade.
        settings: Optional settings override.
    """

    def __init__(self, registry: ModelRegistry, settings: Settings | None = None) -> None:
        self.registry: ModelRegistry = registry
        self.settings: Settings = settings or get_settings()

    def _optimize(
        self, run_id: str, make: Callable[[dict[str, Any] | None, int], ImbPipeline],
        X: pd.DataFrame, y: pd.Series, config: TrainingConfig, progress: ProgressCallback | None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, float]], int]:
        """Run the Optuna study on the training split; log each trial as an MLflow child run."""
        n_jobs_clf = max(1, (os.cpu_count() or 1) // config.optuna_n_jobs)
        try:
            storage: Any = optuna.storages.RDBStorage(
                self.settings.optuna_storage, engine_kwargs={"connect_args": {"timeout": 30}}
            ) if self.settings.optuna_storage.startswith("sqlite") else self.settings.optuna_storage
        except Exception:  # noqa: BLE001 - fall back to in-memory study
            logger.warning("Optuna storage unavailable; using in-memory study.")
            storage = None
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.random_state),
            study_name=f"churn-{config.model_name}-{run_id[:8]}", storage=storage,
        )

        def objective(trial: optuna.Trial) -> float:
            t0 = time.perf_counter()
            params = suggest_params(trial, config.model_name, config.param_space)
            cv = cross_validate_pipeline(
                make(params, n_jobs_clf), X, y, n_splits=config.cv_folds,
                random_state=config.random_state, threshold=config.decision_threshold,
            )
            trial.set_user_attr("cv", cv)
            self.registry.log_trial(
                run_id, trial.number, params, {f"cv_{k}": v["mean"] for k, v in cv.items()},
                time.perf_counter() - t0,
            )
            return cv[config.optimize_metric]["mean"]

        def on_trial(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
            if progress:
                done = len([t for t in st.trials if t.state.is_finished()])
                best = st.best_value if any(t.state == optuna.trial.TrialState.COMPLETE for t in st.trials) else None
                progress({"stage": "optuna", "trials_done": done, "n_trials": config.n_trials, "best_value": best})

        study.optimize(
            objective, n_trials=config.n_trials, timeout=config.timeout_seconds,
            n_jobs=config.optuna_n_jobs, callbacks=[on_trial], catch=(ValueError,),
        )
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise TrainingError("Optuna finished without a single successful trial.")
        best = study.best_trial
        self.registry.log_json(run_id, "optuna_study.json", [
            {"number": t.number, "value": t.value, "params": t.params, "state": t.state.name}
            for t in study.trials
        ])
        return dict(best.params), best.user_attrs["cv"], len(complete)

    def train(
        self, df: pd.DataFrame, config: TrainingConfig, *, job_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> TrainingResult:
        """Validate data, train (baseline or Optuna), evaluate, log and register the model.

        Args:
            df: Raw dataset (real or synthetic) containing the target column.
            config: Training configuration.
            job_id: Optional background-job id stored as a run tag.
            progress: Optional callback receiving progress dictionaries.

        Returns:
            :class:`TrainingResult`.

        Raises:
            DataValidationError: If a data quality gate fails.
            TrainingError: If training cannot complete.
        """
        cfg, started = self.settings, time.perf_counter()
        emit: ProgressCallback = progress or (lambda _: None)
        emit({"stage": "validating"})
        prepared = prepare_training_frame(df, config.target_column, settings=cfg)
        X, y, schema = prepared.X, prepared.y, prepared.schema

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.test_size, stratify=y, random_state=config.random_state
        )
        p = float(y_tr.mean())
        minority_ratio = min(p, 1 - p) / max(p, 1 - p)
        spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

        def make(params: dict[str, Any] | None, n_jobs: int) -> ImbPipeline:
            return build_pipeline(schema, config, params, minority_ratio=minority_ratio,
                                  scale_pos_weight=spw, settings=cfg, n_jobs=n_jobs)

        run_name = config.run_name or f"{config.model_name}-{config.mode}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        tags = {"run_type": "training", "model_name": config.model_name, "mode": config.mode,
                "job_id": job_id or ""}
        with self.registry.run(run_name, tags) as run_id:
            self.registry.log_params(run_id, {
                **{k: v for k, v in config.model_dump().items() if k != "param_space"},
                "param_space": config.param_space, "n_rows": len(X), "n_features": len(schema.feature_columns),
                "minority_ratio_train": round(minority_ratio, 4),
            })
            if config.mode == "optuna":
                best_params, cv_metrics, n_trials = self._optimize(run_id, make, X_tr, y_tr, config, emit)
            else:
                emit({"stage": "cross_validation"})
                best_params, n_trials = {}, 0
                cv_metrics = cross_validate_pipeline(
                    make(None, max(1, os.cpu_count() or 1)), X_tr, y_tr, n_splits=config.cv_folds,
                    random_state=config.random_state, threshold=config.decision_threshold,
                )

            emit({"stage": "final_fit"})
            final = make(best_params or None, max(1, os.cpu_count() or 1)).fit(X_tr, y_tr)
            proba = final.predict_proba(X_te)[:, 1]
            test_metrics = compute_metrics(y_te, proba, config.decision_threshold)
            cm = confusion_matrix(y_te, (proba >= config.decision_threshold).astype(int), labels=[0, 1]).tolist()
            duration = time.perf_counter() - started

            emit({"stage": "logging"})
            self.registry.log_params(run_id, {f"best_{k}": v for k, v in best_params.items()})
            self.registry.log_metrics(run_id, {
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"cv_{k}_mean": v["mean"] for k, v in cv_metrics.items()},
                **{f"cv_{k}_std": v["std"] for k, v in cv_metrics.items()},
                "duration_s": duration, "n_train": len(X_tr), "n_test": len(X_te),
            })
            self.registry.log_figures(run_id, _build_figures(y_te.to_numpy(), proba, cm))
            self.registry.log_pipeline(run_id, final, X_tr)
            self.registry.log_json(run_id, "schema.json", schema.to_dict())
            reference = build_reference_profile(X_tr, schema, random_state=config.random_state)
            reference["predictions"] = build_prediction_profile(proba, config.decision_threshold)
            reference["training_positive_rate"] = float(y_tr.mean())
            self.registry.log_json(run_id, "reference_profile.json", reference)
            self.registry.log_json(run_id, "feature_importance.json", extract_feature_importance(final))
            summary = {
                "model_name": config.model_name, "mode": config.mode, "threshold": config.decision_threshold,
                "test_metrics": test_metrics, "cv_metrics": cv_metrics, "confusion_matrix": cm,
                "best_params": best_params, "config": config.model_dump(), "duration_s": duration,
                "trained_at": datetime.now(timezone.utc).isoformat(), "n_train": len(X_tr), "n_test": len(X_te),
            }
            self.registry.log_json(run_id, "training_summary.json", summary)
            version: int | None = None
            if config.register:
                version = self.registry.register_version(run_id, tags={
                    "model_name": config.model_name, "mode": config.mode,
                    "test_roc_auc": f"{test_metrics['roc_auc']:.4f}", "test_pr_auc": f"{test_metrics['pr_auc']:.4f}",
                })
        emit({"stage": "done", "model_version": version})
        return TrainingResult(
            run_id=run_id, model_name=config.model_name, mode=config.mode, model_version=version,
            test_metrics=test_metrics, cv_metrics=cv_metrics, confusion_matrix=cm, best_params=best_params,
            n_trials=n_trials, threshold=config.decision_threshold, duration_s=duration,
            n_train=len(X_tr), n_test=len(X_te), warnings=prepared.report.warnings,
        )


# --------------------------------------------------------------------------- #
# Background jobs
# --------------------------------------------------------------------------- #
@dataclass
class JobRecord:
    """State of one background training job."""

    job_id: str
    config: dict[str, Any]
    status: Literal["queued", "running", "succeeded", "failed"] = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return dict(self.__dict__)


class TrainingJobManager:
    """Thread-safe registry of training jobs executed off the request thread.

    Instances are owned by the application (e.g. ``app.state``) - no module-level
    mutable state. FastAPI uses :meth:`create_job` + ``BackgroundTasks.add_task(run_job, ...)``;
    scripts/notebooks may call :meth:`submit` to use the internal worker pool.

    Args:
        trainer: Trainer used to execute jobs.
        max_workers: Concurrent training jobs for :meth:`submit`.
    """

    def __init__(self, trainer: ModelTrainer, max_workers: int = 1) -> None:
        self._trainer = trainer
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="train")

    def create_job(self, config: TrainingConfig) -> str:
        """Register a queued job and return its id."""
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = JobRecord(job_id=job_id, config=config.model_dump())
        return job_id

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in changes.items():
                setattr(record, key, value)

    def run_job(self, job_id: str, df: pd.DataFrame, config: TrainingConfig) -> None:
        """Execute a job synchronously (call from a worker thread / background task).

        Failures are captured in the job record instead of propagating.

        Args:
            job_id: Id from :meth:`create_job`.
            df: Training dataset.
            config: Training configuration.
        """
        self._update(job_id, status="running", started_at=datetime.now(timezone.utc).isoformat())
        try:
            result = self._trainer.train(
                df, config, job_id=job_id, progress=lambda p: self._update(job_id, progress=p)
            )
            self._update(job_id, status="succeeded", result=result.to_dict(),
                         finished_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:  # noqa: BLE001 - job boundary: record every failure
            logger.exception("Training job %s failed", job_id)
            self._update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}",
                         finished_at=datetime.now(timezone.utc).isoformat())

    def submit(self, df: pd.DataFrame, config: TrainingConfig) -> str:
        """Create a job and run it on the internal thread pool.

        Returns:
            Job id.
        """
        job_id = self.create_job(config)
        self._pool.submit(self.run_job, job_id, df, config)
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return a snapshot of one job, or ``None`` if unknown."""
        with self._lock:
            record = self._jobs.get(job_id)
            return record.to_dict() if record else None

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return snapshots of all jobs, newest first."""
        with self._lock:
            return [j.to_dict() for j in sorted(self._jobs.values(), key=lambda r: r.created_at, reverse=True)]

    def shutdown(self) -> None:
        """Stop accepting work and wait for running jobs."""
        self._pool.shutdown(wait=True)
