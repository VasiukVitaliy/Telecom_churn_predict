"""FastAPI entrypoint for the telecom churn microservice.

Run with ``uvicorn api.main:app --port 8000`` from ``app_ml_rest_api/``.
All services (registry, trainer, job manager, model service, monitor) are created
in the lifespan handler and stored on ``app.state`` - no module-level mutable state.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.datasets import (
    DatasetNotFoundError, DatasetStore, column_profile, eda_summary, pca_projection,
)
from api.model_service import (
    ModelService, ModelValidationError, NoModelLoadedError, feature_specs,
)
from api.monitoring import InferenceMonitor
from api.schemas import (
    AliasRequest, BatchItem, BatchPredictRequest, BatchPredictResponse, DatasetInfo, HealthResponse,
    PredictRequest, PredictResponse, ReloadResponse, SafeJSONResponse, SensitivityRequest,
    SensitivityResponse, SimulateRequest, SyntheticRequest, TrainRequest, TrainStartResponse,
    ValidateRequest,
)
from ml_core.config import Settings, get_settings
from ml_core.data_generator import generate_synthetic_subscribers
from ml_core.data_validator import (
    ColumnKind, DataValidationError, DatasetSchema, find_target_column, infer_schema,
    load_dataset_bytes, normalize_frame, validate_dataframe,
)
from ml_core.registry import ModelNotFoundError, ModelRegistry, RegistryError
from ml_core.trainer import ModelTrainer, TrainingError, TrainingJobManager

API_VERSION = "1.0.0"
logger = logging.getLogger("churn_api")


@dataclass
class Services:
    """Container of long-lived collaborators (stored on ``app.state``)."""

    settings: Settings
    registry: ModelRegistry
    trainer: ModelTrainer
    jobs: TrainingJobManager
    model_service: ModelService
    monitor: InferenceMonitor
    datasets: DatasetStore
    started_at: float


def svc(request: Request) -> Services:
    """Return the application services from the request."""
    return request.app.state.services  # type: ignore[no-any-return]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory.

    Args:
        settings: Optional settings override (used by tests).

    Returns:
        Configured FastAPI application.

    Example:
        >>> app = create_app()  # doctest: +SKIP
    """
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        cfg.ensure_directories()
        registry = ModelRegistry(cfg)  # verifies DB + artifact storage, raises on failure
        trainer = ModelTrainer(registry, cfg)
        services = Services(
            settings=cfg, registry=registry, trainer=trainer, jobs=TrainingJobManager(trainer),
            model_service=ModelService(registry, cfg), monitor=InferenceMonitor(cfg),
            datasets=DatasetStore(cfg), started_at=time.time(),
        )
        restored = services.monitor.load_history()
        logger.info("Restored %d telemetry rows", restored)
        try:
            await asyncio.to_thread(services.model_service.load, cfg.champion_alias)
        except (ModelNotFoundError, RegistryError, ModelValidationError) as exc:
            logger.warning("Starting without a serving model: %s", exc)
        app.state.services = services
        yield
        services.jobs.shutdown()

    app = FastAPI(
        title="Telecom Churn ML Service", version=API_VERSION, lifespan=lifespan,
        default_response_class=SafeJSONResponse,
    )
    app.state.settings = cfg
    app.add_middleware(
        CORSMiddleware, allow_origins=cfg.cors_origins, allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next: Any) -> Any:
        """Attach the processing time to every response."""
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return response

    # --------------------------------------------------------------- error handlers
    def _err(status: int, detail: str, **extra: Any) -> JSONResponse:
        return SafeJSONResponse(status_code=status, content={"detail": detail, **extra})

    @app.exception_handler(DataValidationError)
    async def _validation(_: Request, exc: DataValidationError) -> JSONResponse:
        return _err(422, str(exc), report=exc.report.to_dict() if exc.report else None)

    @app.exception_handler(ModelValidationError)
    async def _model_validation(_: Request, exc: ModelValidationError) -> JSONResponse:
        return _err(422, str(exc))

    @app.exception_handler(DatasetNotFoundError)
    async def _ds_missing(_: Request, exc: DatasetNotFoundError) -> JSONResponse:
        return _err(404, str(exc))

    @app.exception_handler(ModelNotFoundError)
    async def _model_missing(_: Request, exc: ModelNotFoundError) -> JSONResponse:
        return _err(404, str(exc))

    @app.exception_handler(NoModelLoadedError)
    async def _no_model(_: Request, exc: NoModelLoadedError) -> JSONResponse:
        return _err(503, str(exc))

    @app.exception_handler(RegistryError)
    async def _registry(_: Request, exc: RegistryError) -> JSONResponse:
        return _err(503, str(exc))

    @app.exception_handler(TrainingError)
    async def _training(_: Request, exc: TrainingError) -> JSONResponse:
        return _err(500, str(exc))

    # ------------------------------------------------------------------- health
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        """Liveness + model state (also verifies the MLflow database)."""
        s = svc(request)
        loaded = s.model_service.is_loaded()
        try:
            await asyncio.to_thread(s.registry.verify_connection)
            db_ok = True
        except RegistryError:
            db_ok = False
        model = s.model_service.current() if loaded else None
        return HealthResponse(
            status="ok" if loaded and db_ok else "degraded", model_loaded=loaded,
            model_version=model.version if model else None, model_alias=model.alias if model else None,
            tracking_uri=s.settings.mlflow_tracking_uri, artifact_root=str(s.settings.artifact_root_path),
            uptime_s=round(time.time() - s.started_at, 1), api_version=API_VERSION,
        )

    # ----------------------------------------------------------------- datasets
    @app.get("/datasets", tags=["datasets"])
    async def list_datasets(request: Request) -> list[DatasetInfo]:
        """List stored raw and synthetic datasets."""
        return [DatasetInfo(**d) for d in svc(request).datasets.list()]

    @app.post("/datasets/upload", response_model=DatasetInfo, status_code=201, tags=["datasets"])
    async def upload_dataset(request: Request, file: UploadFile = File(...)) -> DatasetInfo:
        """Upload a real (non-synthetic) CSV / Parquet dataset into ``data/raw``."""
        content = await file.read()
        dataset_id, df = await asyncio.to_thread(svc(request).datasets.save_upload, file.filename or "", content)
        source, name = dataset_id.split("/", 1)
        return DatasetInfo(dataset_id=dataset_id, source=source, filename=name,  # type: ignore[arg-type]
                           size_bytes=len(content), n_rows=len(df), n_columns=df.shape[1])

    @app.post("/datasets/synthetic", response_model=DatasetInfo, status_code=201, tags=["datasets"])
    async def generate_dataset(body: SyntheticRequest, request: Request) -> DatasetInfo:
        """Generate and store a synthetic subscriber dataset."""
        s = svc(request)
        df = await asyncio.to_thread(
            generate_synthetic_subscribers, body.n_samples, body.target_churn_rate,
            body.random_state, body.missing_rate,
        )
        name = body.filename or f"synthetic_{body.n_samples}_{body.random_state}.parquet"
        dataset_id = await asyncio.to_thread(s.datasets.save_frame, df, name)
        size = s.datasets.resolve(dataset_id).stat().st_size
        return DatasetInfo(dataset_id=dataset_id, source="synthetic", filename=name, size_bytes=size,
                           n_rows=len(df), n_columns=df.shape[1])

    @app.post("/datasets/validate", tags=["datasets"])
    async def validate_dataset(body: ValidateRequest, request: Request) -> dict[str, Any]:
        """Infer the schema and run the training quality gates (never raises on failures)."""
        s = svc(request)
        df = await asyncio.to_thread(s.datasets.load, body.dataset_id)
        frame = normalize_frame(df)
        target = find_target_column(frame, body.target_column)
        schema = infer_schema(frame, target, max_cardinality=s.settings.max_categorical_cardinality)
        report = validate_dataframe(frame, schema, mode="training", target=target,
                                    settings=s.settings, raise_on_error=False)
        return {"dataset_id": body.dataset_id, "target": target, "schema": schema.to_dict(),
                "report": report.to_dict(), "feature_columns": schema.feature_columns}

    @app.get("/datasets/{source}/{name}/eda", tags=["datasets"])
    async def dataset_eda(source: str, name: str, request: Request, target: str | None = None) -> dict[str, Any]:
        """Dataset-level EDA summary (schema, missingness, correlations)."""
        s = svc(request)
        df = await asyncio.to_thread(s.datasets.load, f"{source}/{name}")
        return await asyncio.to_thread(eda_summary, df, target, s.settings)

    @app.get("/datasets/{source}/{name}/columns/{column}", tags=["datasets"])
    async def dataset_column(source: str, name: str, column: str, request: Request,
                             target: str | None = None) -> dict[str, Any]:
        """Distribution of one column, split by churn class."""
        df = await asyncio.to_thread(svc(request).datasets.load, f"{source}/{name}")
        return await asyncio.to_thread(column_profile, df, column, target)

    @app.get("/datasets/{source}/{name}/pca", tags=["datasets"])
    async def dataset_pca(source: str, name: str, request: Request, target: str | None = None,
                          n_components: int = Query(3, ge=2, le=10),
                          sample_size: int = Query(3000, ge=100, le=20000)) -> dict[str, Any]:
        """PCA projection of the preprocessed feature space."""
        s = svc(request)
        df = await asyncio.to_thread(s.datasets.load, f"{source}/{name}")
        return await asyncio.to_thread(pca_projection, df, target, s.settings,
                                       n_components=n_components, sample_size=sample_size)

    # ----------------------------------------------------------------- training
    @app.post("/training/start", response_model=TrainStartResponse, status_code=202, tags=["training"])
    async def start_training(body: TrainRequest, background: BackgroundTasks, request: Request) -> TrainStartResponse:
        """Queue a baseline / Optuna training job; returns immediately with a job id."""
        s = svc(request)
        df = await asyncio.to_thread(s.datasets.load, body.dataset_id)
        job_id = s.jobs.create_job(body.config)
        background.add_task(s.jobs.run_job, job_id, df, body.config)
        return TrainStartResponse(job_id=job_id, status="queued")

    @app.get("/training/jobs", tags=["training"])
    async def list_jobs(request: Request) -> list[dict[str, Any]]:
        """All training jobs, newest first."""
        return svc(request).jobs.list_jobs()

    @app.get("/training/jobs/{job_id}", tags=["training"])
    async def get_job(job_id: str, request: Request) -> Any:
        """Status, progress and result of one job."""
        job = svc(request).jobs.get_job(job_id)
        return job if job else _err(404, f"Job '{job_id}' not found.")

    # ------------------------------------------------------------ registry / model
    @app.get("/models/versions", tags=["models"])
    async def model_versions(request: Request) -> list[dict[str, Any]]:
        """Registered versions with aliases and metrics."""
        return await asyncio.to_thread(svc(request).registry.list_model_versions)

    @app.get("/models/runs", tags=["models"])
    async def model_runs(request: Request, limit: int = Query(50, ge=1, le=200)) -> list[dict[str, Any]]:
        """Recent MLflow training runs."""
        return await asyncio.to_thread(svc(request).registry.list_runs, limit)

    @app.post("/models/versions/{version}/alias", tags=["models"])
    async def set_alias(version: int, body: AliasRequest, request: Request) -> dict[str, Any]:
        """Point an alias (e.g. ``champion``, ``challenger``) at a version."""
        await asyncio.to_thread(svc(request).registry.set_alias, body.alias, version)
        return {"alias": body.alias, "version": version}

    @app.post("/models/versions/{version}/promote", tags=["models"])
    async def promote(version: int, request: Request) -> dict[str, Any]:
        """Shortcut: set ``@champion`` to ``version`` (does not reload the API)."""
        s = svc(request)
        await asyncio.to_thread(s.registry.promote, version)
        return {"alias": s.settings.champion_alias, "version": version}

    @app.delete("/models/aliases/{alias}", tags=["models"])
    async def delete_alias(alias: str, request: Request) -> dict[str, Any]:
        """Remove an alias."""
        await asyncio.to_thread(svc(request).registry.delete_alias, alias)
        return {"deleted": alias}

    @app.get("/model/info", tags=["models"])
    async def model_info(request: Request) -> dict[str, Any]:
        """Metadata of the model currently serving traffic."""
        s = svc(request)
        m = s.model_service.current()
        return {"name": m.name, "version": m.version, "alias": m.alias, "run_id": m.run_id,
                "threshold": m.threshold, "loaded_at": m.loaded_at, "summary": m.summary,
                "feature_importance": m.feature_importance, "reload_history": s.model_service.history()}

    @app.get("/model/schema", tags=["models"])
    async def model_schema(request: Request) -> dict[str, Any]:
        """Input feature specification of the serving model (for forms / sensitivity)."""
        m = svc(request).model_service.current()
        return {"version": m.version, "threshold": m.threshold, "features": feature_specs(m)}

    @app.post("/model/reload", response_model=ReloadResponse, tags=["models"])
    async def reload_model(request: Request, alias: str | None = Query(None),
                           version: int | None = Query(None, ge=1)) -> ReloadResponse:
        """Hot-reload the model behind an alias (default ``champion``) without dropping requests."""
        s = svc(request)
        event = await asyncio.to_thread(
            s.model_service.load, alias or (None if version else s.settings.champion_alias), version
        )
        return ReloadResponse(reloaded=True, model_version=event["model_version"], model_alias=event["model_alias"],
                              previous_version=event["previous_version"], duration_ms=event["duration_ms"])

    # --------------------------------------------------------------- prediction
    @app.post("/predict", response_model=PredictResponse, tags=["prediction"])
    async def predict(body: PredictRequest, background: BackgroundTasks, request: Request) -> PredictResponse:
        """Score one subscriber; the request is logged to telemetry in the background."""
        s = svc(request)
        started = time.perf_counter()
        model = s.model_service.current()
        frame, warns = s.model_service.prepare([body.features], model, single=True)
        proba = await asyncio.to_thread(s.model_service.score, model, frame)
        threshold = body.threshold or model.threshold
        latency = (time.perf_counter() - started) * 1000
        rid = uuid.uuid4().hex[:12]
        label = int(proba[0] >= threshold)
        background.add_task(
            s.monitor.record, request_id=rid, model_version=model.version, latency_ms=latency,
            features=frame.to_dict("records"), probabilities=[float(proba[0])], labels=[label],
        )
        return PredictResponse(request_id=rid, churn_probability=float(proba[0]), prediction=label,
                               threshold=threshold, model_version=model.version, model_alias=model.alias,
                               latency_ms=round(latency, 3), warnings=warns)

    async def _score_batch(s: Services, records: pd.DataFrame | list[dict[str, Any]], threshold: float | None,
                           background: BackgroundTasks) -> BatchPredictResponse:
        started = time.perf_counter()
        model = s.model_service.current()
        frame, warns = s.model_service.prepare(records, model, single=False)
        proba = await asyncio.to_thread(s.model_service.score, model, frame)
        thr = threshold or model.threshold
        labels = (proba >= thr).astype(int)
        latency = (time.perf_counter() - started) * 1000
        rid = uuid.uuid4().hex[:12]
        background.add_task(
            s.monitor.record, request_id=rid, model_version=model.version, latency_ms=latency,
            features=frame.to_dict("records"), probabilities=proba.tolist(), labels=labels.tolist(),
        )
        return BatchPredictResponse(
            request_id=rid, n=len(frame), positive_rate=float(labels.mean()), threshold=thr,
            model_version=model.version, model_alias=model.alias, latency_ms=round(latency, 3),
            predictions=[BatchItem(index=i, churn_probability=float(p), prediction=int(l))
                         for i, (p, l) in enumerate(zip(proba, labels))],
            warnings=warns,
        )

    @app.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
    async def predict_batch(body: BatchPredictRequest, background: BackgroundTasks,
                            request: Request) -> BatchPredictResponse:
        """Score many subscribers (all schema features required; quality gates enforced)."""
        return await _score_batch(svc(request), body.records, body.threshold, background)

    @app.post("/predict/batch/upload", response_model=BatchPredictResponse, tags=["prediction"])
    async def predict_upload(background: BackgroundTasks, request: Request, file: UploadFile = File(...),
                             threshold: float | None = Query(None, gt=0.0, lt=1.0)) -> BatchPredictResponse:
        """Score a CSV / Parquet file."""
        content = await file.read()
        df = await asyncio.to_thread(load_dataset_bytes, content, file.filename or "upload.csv")
        return await _score_batch(svc(request), df, threshold, background)

    @app.post("/predict/sensitivity", response_model=SensitivityResponse, tags=["prediction"])
    async def sensitivity(body: SensitivityRequest, request: Request) -> SensitivityResponse:
        """Vary one feature over a grid (numeric range or categories) around a base record."""
        s = svc(request)
        model = s.model_service.current()
        specs = {sp["name"]: sp for sp in feature_specs(model)}
        if body.feature not in specs:
            raise DataValidationError(f"Unknown feature '{body.feature}'.")
        spec = specs[body.feature]
        base = {name: sp["default"] for name, sp in specs.items()}
        unknown = [k for k in body.features if k not in specs]
        if unknown:
            raise DataValidationError(f"Unknown fields in base record: {unknown}")
        base.update(body.features)
        if body.values:
            grid: list[Any] = body.values
        elif spec["kind"] == "numeric":
            lo, hi = spec["min"], spec["max"]
            if lo is None or hi is None or lo == hi:
                raise DataValidationError(f"Feature '{body.feature}' has no usable numeric range.")
            grid = np.linspace(lo, hi, body.n_points).round(4).tolist()
        else:
            grid = list(spec["categories"])
        rows = pd.DataFrame([{**base, body.feature: v} for v in grid])
        proba = await asyncio.to_thread(s.model_service.score, model, rows)
        base_p = await asyncio.to_thread(s.model_service.score, model, pd.DataFrame([base]))
        return SensitivityResponse(feature=body.feature, kind=spec["kind"], values=grid,
                                   probabilities=proba.tolist(), base_probability=float(base_p[0]),
                                   threshold=body.threshold or model.threshold, model_version=model.version)

    # --------------------------------------------------------------- monitoring
    def _baseline(model_ref: dict[str, Any]) -> float | None:
        return model_ref.get("predictions", {}).get("expected_positive_rate")

    @app.get("/monitoring/summary", tags=["monitoring"])
    async def monitoring_summary(request: Request, all_versions: bool = False) -> dict[str, Any]:
        """Request volumes, latency and live churn rate vs the training baseline."""
        s = svc(request)
        m = s.model_service.current()
        return {"model_version": m.version, **s.monitor.summary(
            _baseline(m.reference_profile), None if all_versions else m.version)}

    @app.get("/monitoring/timeseries", tags=["monitoring"])
    async def monitoring_timeseries(request: Request, bucket_seconds: int = Query(60, ge=5, le=86400),
                                    n_buckets: int = Query(60, ge=1, le=500)) -> dict[str, Any]:
        """Bucketed volumes and rolling churn rate for live charts."""
        s = svc(request)
        m = s.model_service.current()
        return {"baseline_rate": _baseline(m.reference_profile), "bucket_seconds": bucket_seconds,
                "buckets": s.monitor.timeseries(bucket_seconds, n_buckets)}

    @app.get("/monitoring/drift", tags=["monitoring"])
    async def monitoring_drift(request: Request, min_samples: int = Query(30, ge=5, le=5000)) -> dict[str, Any]:
        """PSI / KS feature drift and prediction distribution shift for the serving model."""
        s = svc(request)
        m = s.model_service.current()
        return await asyncio.to_thread(s.monitor.drift, m.reference_profile, m.version, min_samples)

    @app.get("/monitoring/recent", tags=["monitoring"])
    async def monitoring_recent(request: Request, n: int = Query(200, ge=1, le=5000)) -> list[dict[str, Any]]:
        """Most recent scored rows (probabilities, labels, latency)."""
        return svc(request).monitor.recent(n)

    @app.post("/monitoring/reset", tags=["monitoring"])
    async def monitoring_reset(request: Request) -> dict[str, str]:
        """Clear the in-memory telemetry window."""
        svc(request).monitor.reset()
        return {"status": "reset"}

    @app.post("/monitoring/simulate", tags=["monitoring"])
    async def simulate_traffic(body: SimulateRequest, background: BackgroundTasks, request: Request) -> dict[str, Any]:
        """Demo helper: score synthetic subscribers (optionally drifted) to populate dashboards."""
        s = svc(request)
        model = s.model_service.current()
        schema = DatasetSchema.from_dict(model.schema)
        seed = body.random_state if body.random_state is not None else int(time.time()) % 100_000
        df = await asyncio.to_thread(generate_synthetic_subscribers, max(body.n, 500), 0.2, seed)
        missing = [c for c in schema.feature_columns if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"Simulation needs a model trained on the synthetic schema; missing columns: {missing[:5]}"
            )
        df = df.head(body.n).copy()
        if body.drift:
            numeric = [c for c in schema.numeric_columns if c in df.columns][:4]
            for col in numeric:
                df[col] = df[col] * 1.6 + 1
            if "contract_type" in df.columns:
                df["contract_type"] = "Month-to-month"
        result = await _score_batch(s, df[schema.feature_columns], None, background)
        return {"simulated": result.n, "drift": body.drift, "positive_rate": result.positive_rate,
                "model_version": result.model_version}

    return app


app: FastAPI = create_app()
