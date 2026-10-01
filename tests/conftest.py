"""Shared fixtures.

All state (SQLite DBs, artifacts, datasets, telemetry) lives in a temporary directory,
so the suite never touches the real ``data/`` or ``models/`` folders. A single tiny model
pair is trained once per session and reused by the API and hot-reload tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import create_app  # noqa: E402
from ml_core.config import Settings  # noqa: E402
from ml_core.data_generator import generate_synthetic_subscribers  # noqa: E402

FAST_TRAIN: dict[str, Any] = {"cv_folds": 2, "test_size": 0.25, "random_state": 42}


def make_settings(base: Path, **overrides: Any) -> Settings:
    """Build isolated settings rooted in ``base``.

    Args:
        base: Temporary directory.
        **overrides: Extra :class:`Settings` fields.

    Returns:
        Settings with absolute SQLite / artifact paths under ``base``.
    """
    models = base / "models"
    params: dict[str, Any] = {
        "data_raw_dir": base / "raw", "data_synthetic_dir": base / "synthetic", "models_dir": models,
        "mlflow_tracking_uri": f"sqlite:///{(models / 'mlflow.db').as_posix()}",
        "mlflow_artifact_root": str(models / "artifacts"),
        "optuna_storage": f"sqlite:///{(models / 'optuna.db').as_posix()}",
        "min_rows": 200, "min_minority_samples": 30, "monitor_window_size": 2000,
    }
    params.update(overrides)
    return Settings(**params)


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Small deterministic synthetic dataset (1,500 rows)."""
    return generate_synthetic_subscribers(1500, 0.2, 42)


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    """Function-scoped isolated settings."""
    return make_settings(tmp_path)


@pytest.fixture(scope="session")
def trained_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """App with two registered versions (LightGBM v1, RandomForest v2); ``@champion`` -> v1.

    Starlette's ``TestClient`` runs background tasks before returning, so each
    ``POST /training/start`` completes synchronously here.
    """
    settings = make_settings(tmp_path_factory.mktemp("churn"))
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post("/datasets/synthetic", json={"n_samples": 1200, "filename": "ci.csv"})
        assert created.status_code == 201, created.text
        dataset_id = created.json()["dataset_id"]
        client.app.state.test_dataset_id = dataset_id
        for model in ("lightgbm", "random_forest"):
            cfg = {**FAST_TRAIN, "model_name": model, "mode": "baseline"}
            r = client.post("/training/start", json={"dataset_id": dataset_id, "config": cfg})
            assert r.status_code == 202, r.text
            job = client.get(f"/training/jobs/{r.json()['job_id']}").json()
            assert job["status"] == "succeeded", job.get("error")
        assert client.post("/models/versions/1/promote").status_code == 200
        assert client.post("/model/reload").status_code == 200
        yield client


@pytest.fixture()
def empty_client(tmp_path: Path) -> Iterator[TestClient]:
    """App with a fresh registry and no model loaded."""
    with TestClient(create_app(make_settings(tmp_path))) as client:
        yield client


def feature_record(client: TestClient) -> dict[str, Any]:
    """A valid single-subscriber record built from ``GET /model/schema`` defaults."""
    specs = client.get("/model/schema").json()["features"]
    return {s["name"]: s["default"] for s in specs}
