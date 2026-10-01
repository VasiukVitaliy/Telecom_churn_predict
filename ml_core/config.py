"""Central, environment-driven configuration for the churn ML system.

All filesystem locations are resolved relative to the project root
(``app_ml_rest_api/``) so that the code behaves identically regardless of the
current working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_FILE: Path = PROJECT_ROOT / ".env"
_SQLITE_PREFIX: str = "sqlite:///"


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a possibly relative path against ``root``.

    Args:
        path: Absolute or relative filesystem path.
        root: Base directory used for relative paths.

    Returns:
        Absolute, normalised path.

    Example:
        >>> resolve_path("data/raw").is_absolute()
        True
    """
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def resolve_db_uri(uri: str, root: Path = PROJECT_ROOT) -> str:
    """Make relative SQLite URIs absolute; leave every other URI untouched.

    Args:
        uri: SQLAlchemy-style database URI (``sqlite:///models/mlflow.db``,
            ``postgresql+psycopg2://...``).
        root: Base directory for relative SQLite paths.

    Returns:
        URI with an absolute SQLite path when applicable.

    Example:
        >>> resolve_db_uri("sqlite:///models/x.db").startswith("sqlite:////")
        True
    """
    if uri.startswith(_SQLITE_PREFIX) and not uri.startswith(_SQLITE_PREFIX + "/"):
        rel = uri[len(_SQLITE_PREFIX):]
        if rel and rel != ":memory:" and not Path(rel).is_absolute():
            return f"{_SQLITE_PREFIX}{(root / rel).resolve().as_posix()}"
    return uri


class Settings(BaseSettings):
    """Application settings loaded from environment variables / ``.env``.

    Attributes:
        mlflow_tracking_uri: Relational DB URI for MLflow runs + model registry.
        mlflow_artifact_root: Directory that stores serialized artifacts.
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE), env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    app_env: Literal["development", "testing", "production"] = "development"
    random_state: int = 42

    # --- paths -------------------------------------------------------------
    data_raw_dir: Path = Path("data/raw")
    data_synthetic_dir: Path = Path("data/synthetic")
    models_dir: Path = Path("models")

    # --- MLflow ------------------------------------------------------------
    mlflow_tracking_uri: str = "sqlite:///models/mlflow.db"
    mlflow_artifact_root: str = "./models/artifacts/"
    mlflow_experiment_name: str = "telecom-churn"
    registered_model_name: str = "TelecomChurnClassifier"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"

    # --- data quality gates ------------------------------------------------
    max_missing_fraction: float = Field(0.30, ge=0.0, le=1.0)
    min_rows: int = Field(200, ge=10)
    min_minority_samples: int = Field(50, ge=5)
    max_categorical_cardinality: int = Field(50, ge=2)
    non_negative_patterns: list[str] = Field(
        default_factory=lambda: [
            "tenure", "charge", "spend", "minutes", "sms", r"(^|_)gb(_|$)",
            "calls", "fee", "amount", "months", "days", r"(^|_)age$", r"_count$",
        ]
    )

    # --- preprocessing / training defaults --------------------------------
    numeric_imputation: Literal["median", "mean", "constant", "iterative"] = "median"
    categorical_imputation: Literal["most_frequent", "constant"] = "most_frequent"
    sampling_strategy: Literal["none", "class_weight", "random_over", "smote"] = "class_weight"
    sampling_ratio: float = Field(0.6, gt=0.0, le=1.0)
    test_size: float = Field(0.2, gt=0.0, lt=0.5)
    cv_folds: int = Field(5, ge=2, le=20)
    optuna_storage: str = "sqlite:///models/optuna.db"
    optuna_n_jobs: int = Field(1, ge=1)

    # --- monitoring --------------------------------------------------------
    monitor_window_size: int = Field(5000, ge=100)
    psi_warning: float = 0.10
    psi_alert: float = 0.25

    # --- API / frontend ----------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])
    api_base_url: str = "http://localhost:8000"

    @model_validator(mode="after")
    def _resolve_locations(self) -> "Settings":
        """Turn every relative path / SQLite URI into an absolute one."""
        self.data_raw_dir = resolve_path(self.data_raw_dir)
        self.data_synthetic_dir = resolve_path(self.data_synthetic_dir)
        self.models_dir = resolve_path(self.models_dir)
        self.mlflow_tracking_uri = resolve_db_uri(self.mlflow_tracking_uri)
        self.optuna_storage = resolve_db_uri(self.optuna_storage)
        self.mlflow_artifact_root = str(resolve_path(self.mlflow_artifact_root))
        return self

    @property
    def artifact_root_path(self) -> Path:
        """Absolute artifact root directory."""
        return Path(self.mlflow_artifact_root)

    @property
    def artifact_root_uri(self) -> str:
        """Artifact root as a ``file://`` URI (as required by MLflow)."""
        return self.artifact_root_path.as_uri()

    @property
    def model_cache_dir(self) -> Path:
        """Local directory where registry model versions are downloaded."""
        return self.models_dir / "cache"

    def ensure_directories(self) -> None:
        """Create every directory the system writes to (idempotent).

        Raises:
            OSError: If a directory cannot be created.
        """
        for directory in (
            self.data_raw_dir, self.data_synthetic_dir, self.models_dir,
            self.artifact_root_path, self.model_cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for uri in (self.mlflow_tracking_uri, self.optuna_storage):
            if uri.startswith(_SQLITE_PREFIX) and ":memory:" not in uri:
                Path(uri[len(_SQLITE_PREFIX):]).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process-wide settings (read-only configuration).

    Tests may build ``Settings(...)`` directly or call
    ``get_settings.cache_clear()`` after changing the environment.

    Returns:
        Validated :class:`Settings` instance.
    """
    return Settings()
