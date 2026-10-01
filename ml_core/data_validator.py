"""Dataset ingestion, dynamic schema inference, quality gates and drift statistics.

Responsibilities:
    * Load CSV / Parquet files (from disk or raw bytes).
    * Infer column roles (numeric / categorical / id / constant / datetime / target).
    * Enforce quality gates (missingness threshold, schema conformity, ranges).
    * Build a *reference profile* of the training data and compute PSI / KS drift
      against it (used by live monitoring).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ml_core.config import Settings, get_settings

TARGET_CANDIDATES: tuple[str, ...] = ("churn", "is_churn", "churned", "target", "label")
_POSITIVE = {"yes", "y", "true", "1", "churn", "churned", "1.0"}
_NEGATIVE = {"no", "n", "false", "0", "retained", "stayed", "0.0"}
_ID_NAME = re.compile(
    r"(id|.*[_\s\-]id|customer_?id|subscriber_?id|.*uuid|.*guid)", re.IGNORECASE
)
_SIGNED_HINT = re.compile(r"change|trend|delta|diff|growth", re.IGNORECASE)
OTHER_BUCKET = "__other__"
MISSING_BUCKET = "__missing__"


class DataValidationError(Exception):
    """Raised when a dataset fails a strict quality gate.

    Attributes:
        report: Optional :class:`ValidationReport` with all detected issues.
    """

    def __init__(self, message: str, report: "ValidationReport | None" = None) -> None:
        super().__init__(message)
        self.report: ValidationReport | None = report


class ColumnKind(str, Enum):
    """Semantic role of a dataset column."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    ID = "id"
    CONSTANT = "constant"
    DATETIME = "datetime"
    TARGET = "target"


@dataclass
class ColumnSchema:
    """Inferred description of a single column."""

    name: str
    kind: ColumnKind
    dtype: str
    n_unique: int
    missing_pct: float
    min_value: float | None = None
    max_value: float | None = None
    categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "name": self.name, "kind": self.kind.value, "dtype": self.dtype,
            "n_unique": self.n_unique, "missing_pct": self.missing_pct,
            "min_value": self.min_value, "max_value": self.max_value,
            "categories": list(self.categories),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnSchema":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            name=data["name"], kind=ColumnKind(data["kind"]), dtype=data["dtype"],
            n_unique=int(data["n_unique"]), missing_pct=float(data["missing_pct"]),
            min_value=data.get("min_value"), max_value=data.get("max_value"),
            categories=list(data.get("categories", [])),
        )


@dataclass
class DatasetSchema:
    """Schema of a whole dataset."""

    columns: dict[str, ColumnSchema]
    target: str | None
    n_rows: int

    def _of_kind(self, kind: ColumnKind) -> list[str]:
        return [c.name for c in self.columns.values() if c.kind == kind]

    @property
    def numeric_columns(self) -> list[str]:
        """Names of continuous numeric features."""
        return self._of_kind(ColumnKind.NUMERIC)

    @property
    def categorical_columns(self) -> list[str]:
        """Names of categorical features."""
        return self._of_kind(ColumnKind.CATEGORICAL)

    @property
    def id_columns(self) -> list[str]:
        """Names of identifier columns (excluded from modelling)."""
        return self._of_kind(ColumnKind.ID)

    @property
    def feature_columns(self) -> list[str]:
        """Model input columns (numeric + categorical), in dataset order."""
        usable = {ColumnKind.NUMERIC, ColumnKind.CATEGORICAL}
        return [c.name for c in self.columns.values() if c.kind in usable]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "target": self.target, "n_rows": self.n_rows,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetSchema":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            columns={k: ColumnSchema.from_dict(v) for k, v in data["columns"].items()},
            target=data.get("target"), n_rows=int(data.get("n_rows", 0)),
        )


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_dataframe`."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_pct: dict[str, float] = field(default_factory=dict)
    n_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "passed": self.passed, "errors": self.errors, "warnings": self.warnings,
            "missing_pct": self.missing_pct, "n_rows": self.n_rows,
        }


@dataclass
class PreparedData:
    """Validated training data ready for splitting."""

    X: pd.DataFrame
    y: pd.Series
    schema: DatasetSchema
    report: ValidationReport


# --------------------------------------------------------------------------- #
# Loading & normalisation
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV/TSV/Parquet file.

    Args:
        path: File location.

    Returns:
        DataFrame with stripped column names.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataValidationError: If the extension is unsupported or the file is unreadable/empty.

    Example:
        >>> df = load_dataset("data/raw/telco.csv")  # doctest: +SKIP
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return load_dataset_bytes(file_path.read_bytes(), file_path.name)


def load_dataset_bytes(content: bytes, filename: str) -> pd.DataFrame:
    """Load a dataset from raw bytes (used for API uploads).

    Args:
        content: File content.
        filename: Original file name; its suffix selects the parser.

    Returns:
        DataFrame with stripped column names.

    Raises:
        DataValidationError: If the format is unsupported, unreadable or empty.
    """
    suffix = Path(filename).suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            df = pd.read_csv(io.BytesIO(content))
        elif suffix == ".tsv":
            df = pd.read_csv(io.BytesIO(content), sep="\t")
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(io.BytesIO(content))
        else:
            raise DataValidationError(f"Unsupported file type '{suffix}'. Use CSV or Parquet.")
    except DataValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any parser failure uniformly
        raise DataValidationError(f"Could not parse '{filename}': {exc}") from exc
    if df.empty:
        raise DataValidationError(f"Dataset '{filename}' is empty.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where blank / whitespace-only strings become ``NaN``.

    Args:
        df: Raw dataframe.

    Returns:
        Normalised copy (input is never mutated).
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].replace(r"^\s*$", np.nan, regex=True)
    return out


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to float, turning unparsable entries into ``NaN``."""
    return pd.to_numeric(series, errors="coerce")


def find_target_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    """Locate the churn column (case-insensitive).

    Args:
        df: Dataset.
        preferred: Explicit target column name; wins when present.

    Returns:
        Actual column name in ``df``.

    Raises:
        DataValidationError: If no target column can be found.
    """
    lookup = {c.lower(): c for c in df.columns}
    if preferred:
        if preferred.lower() in lookup:
            return lookup[preferred.lower()]
        raise DataValidationError(f"Target column '{preferred}' not found in dataset.")
    for name in TARGET_CANDIDATES:
        if name in lookup:
            return lookup[name]
    raise DataValidationError(
        f"No target column found; expected one of {TARGET_CANDIDATES} or an explicit name."
    )


def encode_target(series: pd.Series) -> pd.Series:
    """Encode Yes/No/True/False/1/0 labels to floats (1.0 = churn); unknown -> NaN.

    Args:
        series: Raw target column.

    Returns:
        Float series with values in {0.0, 1.0, NaN}.

    Raises:
        DataValidationError: If any non-null label cannot be interpreted.
    """
    text = series.astype("string").str.strip().str.lower()
    encoded = text.map(
        lambda v: np.nan if pd.isna(v) else 1.0 if v in _POSITIVE else 0.0 if v in _NEGATIVE else np.nan
    )
    unknown = series.notna() & encoded.isna()
    if bool(unknown.any()):
        bad = sorted(set(series[unknown].astype(str)))[:5]
        raise DataValidationError(f"Unrecognised target labels: {bad}")
    return encoded.astype("float64")


# --------------------------------------------------------------------------- #
# Schema inference
# --------------------------------------------------------------------------- #
def _looks_like_id(name: str, series: pd.Series, n_unique: int, n_non_null: int) -> bool:
    if _ID_NAME.fullmatch(name.strip()):
        return True
    if pd.api.types.is_float_dtype(series) or n_non_null < 100:
        return False
    if not pd.api.types.is_numeric_dtype(series):
        parsed = _coerce_numeric(series.dropna())  # numeric-looking strings are measurements, not ids
        if len(parsed) and parsed.notna().mean() >= 0.95:
            return False
    return n_unique / max(n_non_null, 1) >= 0.98 and not pd.api.types.is_bool_dtype(series)


def infer_schema(
    df: pd.DataFrame, target: str | None = None, *, max_cardinality: int = 50
) -> DatasetSchema:
    """Infer column roles, dtypes and cardinality from a dataframe.

    Blank-string numerics (e.g. Telco ``TotalCharges``) are detected and typed as numeric.

    Args:
        df: Dataset (raw or normalised).
        target: Name of the target column, if known.
        max_cardinality: Max categories stored per categorical column.

    Returns:
        :class:`DatasetSchema` describing every column.

    Example:
        >>> schema = infer_schema(pd.DataFrame({"a": [1.5, 2.5, 3.5], "b": ["x", "y", "x"]}))
        >>> schema.numeric_columns, schema.categorical_columns
        (['a'], ['b'])
    """
    frame = normalize_frame(df)
    columns: dict[str, ColumnSchema] = {}
    for col in frame.columns:
        s = frame[col]
        non_null = s.dropna()
        n_unique = int(non_null.nunique())
        min_v: float | None = None
        max_v: float | None = None
        cats: list[str] = []
        if col == target:
            kind = ColumnKind.TARGET
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = ColumnKind.DATETIME
        elif n_unique <= 1:
            kind = ColumnKind.CONSTANT
        elif _looks_like_id(col, s, n_unique, len(non_null)):
            kind = ColumnKind.ID
        elif pd.api.types.is_bool_dtype(s):
            kind = ColumnKind.CATEGORICAL
        elif pd.api.types.is_numeric_dtype(s):
            kind = ColumnKind.CATEGORICAL if n_unique <= 2 else ColumnKind.NUMERIC
        else:
            parsed = _coerce_numeric(non_null)
            numeric_like = len(non_null) > 0 and parsed.notna().mean() >= 0.95
            kind = (
                ColumnKind.NUMERIC if numeric_like and parsed.nunique() > 2
                else ColumnKind.CATEGORICAL
            )
        if kind == ColumnKind.NUMERIC:
            values = _coerce_numeric(non_null)
            if len(values):
                min_v, max_v = float(values.min()), float(values.max())
        if kind == ColumnKind.CATEGORICAL:
            cats = [str(v) for v in non_null.astype(str).value_counts().index[:max_cardinality]]
        columns[col] = ColumnSchema(
            name=col, kind=kind, dtype=str(s.dtype), n_unique=n_unique,
            missing_pct=float(s.isna().mean()), min_value=min_v, max_value=max_v, categories=cats,
        )
    return DatasetSchema(columns=columns, target=target, n_rows=len(frame))


# --------------------------------------------------------------------------- #
# Quality gates
# --------------------------------------------------------------------------- #
def validate_missingness(
    df: pd.DataFrame, columns: Sequence[str], max_fraction: float
) -> tuple[dict[str, float], list[str]]:
    """Compute per-column missing percentages and list threshold violations.

    Args:
        df: Normalised dataframe.
        columns: Columns to check.
        max_fraction: Maximum tolerated missing fraction (0-1).

    Returns:
        ``(missing_pct_by_column, violation_messages)``.
    """
    present = [c for c in columns if c in df.columns]
    pct = {c: round(float(df[c].isna().mean()) * 100, 3) for c in present}
    violations = [
        f"Column '{c}' has {p:.1f}% missing values (limit {max_fraction * 100:.1f}%)."
        for c, p in pct.items() if p / 100 > max_fraction
    ]
    return pct, violations


def validate_dataframe(
    df: pd.DataFrame,
    schema: DatasetSchema | None = None,
    *,
    mode: Literal["training", "inference"] = "training",
    target: str | None = None,
    settings: Settings | None = None,
    check_missingness: bool = True,
    strict_extra_columns: bool = False,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Run all quality gates on a dataset.

    Gates: minimum size, required column presence, unexpected columns, dtype
    conformity, per-column missingness threshold, non-negative range rules,
    training-range plausibility (warning) and target class balance (training).

    Args:
        df: Dataset to validate.
        schema: Expected schema. When ``None`` it is inferred (training only).
        mode: ``"training"`` requires a target; ``"inference"`` requires a schema.
        target: Target column name (training). Auto-detected if omitted.
        settings: Settings supplying thresholds; defaults to :func:`get_settings`.
        check_missingness: Disable for single-record inference where a missing
            field is legitimately imputed.
        strict_extra_columns: Treat unexpected columns as errors instead of warnings.
        raise_on_error: Raise :class:`DataValidationError` when any error is found.

    Returns:
        The :class:`ValidationReport`.

    Raises:
        DataValidationError: If ``raise_on_error`` and at least one gate fails.
    """
    cfg = settings or get_settings()
    report = ValidationReport(n_rows=len(df))
    frame = normalize_frame(df)

    if mode == "inference" and schema is None:
        raise DataValidationError("Inference validation requires an expected schema.")
    if len(frame) == 0:
        report.errors.append("Dataset has no rows.")
    if mode == "training" and len(frame) < cfg.min_rows:
        report.errors.append(f"Dataset has {len(frame)} rows; at least {cfg.min_rows} required.")

    target_name: str | None = None
    if mode == "training":
        try:
            target_name = find_target_column(frame, target or (schema.target if schema else None))
        except DataValidationError as exc:
            report.errors.append(str(exc))
        if schema is None:
            schema = infer_schema(frame, target_name, max_cardinality=cfg.max_categorical_cardinality)

    if schema is not None:
        features = schema.feature_columns
        missing_cols = [c for c in features if c not in frame.columns]
        if missing_cols:
            report.errors.append(f"Missing required columns: {missing_cols}")
        known = set(schema.columns) | ({target_name} if target_name else set())
        extra = [c for c in frame.columns if c not in known]
        if extra:
            (report.errors if strict_extra_columns else report.warnings).append(
                f"Unexpected columns: {extra}"
            )
        present = [c for c in features if c in frame.columns]

        if check_missingness:
            report.missing_pct, violations = validate_missingness(
                frame, present, cfg.max_missing_fraction
            )
            report.errors.extend(violations)

        patterns = [re.compile(p, re.IGNORECASE) for p in cfg.non_negative_patterns]
        for col in present:
            spec = schema.columns[col]
            if spec.kind == ColumnKind.NUMERIC:
                raw = frame[col]
                values = _coerce_numeric(raw)
                bad_type = int((raw.notna() & values.isna()).sum())
                if bad_type:
                    report.errors.append(
                        f"Column '{col}' expected numeric but {bad_type} value(s) are non-numeric."
                    )
                if any(p.search(col) for p in patterns) and not _SIGNED_HINT.search(col):
                    negatives = int((values < 0).sum())
                    if negatives:
                        report.errors.append(f"Column '{col}' has {negatives} negative value(s).")
                if spec.min_value is not None and spec.max_value is not None and values.notna().any():
                    span = max(spec.max_value - spec.min_value, 1e-9)
                    outside = ((values < spec.min_value - span) | (values > spec.max_value + span)).mean()
                    if outside > 0.05:
                        report.warnings.append(
                            f"Column '{col}': {outside:.1%} of values lie far outside the training range."
                        )
            elif spec.kind == ColumnKind.CATEGORICAL and spec.categories:
                seen = set(frame[col].dropna().astype(str))
                unseen = sorted(seen - set(spec.categories))
                if unseen:
                    report.warnings.append(f"Column '{col}' has unseen categories: {unseen[:5]}")

    if mode == "training" and target_name is not None:
        try:
            y = encode_target(frame[target_name]).dropna()
            counts = y.value_counts()
            if len(counts) < 2:
                report.errors.append("Target has a single class; cannot train a classifier.")
            elif int(counts.min()) < cfg.min_minority_samples:
                report.errors.append(
                    f"Minority class has {int(counts.min())} samples; "
                    f"at least {cfg.min_minority_samples} required."
                )
            if len(y) < len(frame):
                report.warnings.append(f"{len(frame) - len(y)} rows with missing target will be dropped.")
        except DataValidationError as exc:
            report.errors.append(str(exc))

    report.passed = not report.errors
    if raise_on_error and not report.passed:
        raise DataValidationError("; ".join(report.errors), report)
    return report


def prepare_training_frame(
    df: pd.DataFrame, target: str | None = None, *, settings: Settings | None = None
) -> PreparedData:
    """Validate a raw dataset and split it into features ``X`` and binary target ``y``.

    Args:
        df: Raw dataset (real or synthetic).
        target: Target column; auto-detected when ``None``.
        settings: Optional settings override.

    Returns:
        :class:`PreparedData` with schema-selected feature columns only.

    Raises:
        DataValidationError: If any quality gate fails.
    """
    cfg = settings or get_settings()
    frame = normalize_frame(df)
    target_name = find_target_column(frame, target)
    schema = infer_schema(frame, target_name, max_cardinality=cfg.max_categorical_cardinality)
    report = validate_dataframe(frame, schema, mode="training", target=target_name, settings=cfg)
    y_all = encode_target(frame[target_name])
    keep = y_all.notna()
    X = frame.loc[keep, schema.feature_columns].reset_index(drop=True)
    y = y_all[keep].astype(int).reset_index(drop=True)
    return PreparedData(X=X, y=y, schema=schema, report=report)


# --------------------------------------------------------------------------- #
# Reference profile & drift statistics
# --------------------------------------------------------------------------- #
def population_stability_index(expected: Sequence[float], actual: Sequence[float], eps: float = 1e-4) -> float:
    """Population Stability Index between two binned distributions.

    Args:
        expected: Reference bin proportions.
        actual: Live bin proportions (same bins).
        eps: Floor to avoid ``log(0)``.

    Returns:
        PSI value (``<0.1`` stable, ``0.1-0.25`` moderate shift, ``>0.25`` major shift).
    """
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference_profile(
    X: pd.DataFrame, schema: DatasetSchema, *, n_bins: int = 10, max_sample: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Summarise the training feature distributions for later drift comparison.

    Args:
        X: Training features (raw values).
        schema: Dataset schema.
        n_bins: Number of quantile bins for numeric features.
        max_sample: Values retained per numeric feature for KS tests.
        random_state: Sampling seed.

    Returns:
        JSON-serialisable profile ``{"features": {...}, "n_rows": int}``.
    """
    rng = np.random.default_rng(random_state)
    features: dict[str, Any] = {}
    for col in schema.numeric_columns:
        values = _coerce_numeric(X[col]).dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        inner = edges[1:-1]
        counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
        sample = values if len(values) <= max_sample else rng.choice(values, max_sample, replace=False)
        features[col] = {
            "kind": "numeric", "inner_edges": inner.tolist(),
            "expected": (counts / counts.sum()).tolist(),
            "mean": float(values.mean()), "std": float(values.std()),
            "missing_pct": float(X[col].isna().mean() * 100),
            "sample": np.round(sample, 6).tolist(),
        }
    for col in schema.categorical_columns:
        cats = schema.columns[col].categories
        s = X[col].astype("object").where(X[col].notna(), MISSING_BUCKET).astype(str)
        freq = s.value_counts(normalize=True)
        expected = {k: float(freq.get(k, 0.0)) for k in cats}
        if MISSING_BUCKET in freq:
            expected[MISSING_BUCKET] = float(freq[MISSING_BUCKET])
        expected[OTHER_BUCKET] = max(0.0, 1.0 - sum(expected.values()))
        features[col] = {"kind": "categorical", "expected": expected,
                         "missing_pct": float(X[col].isna().mean() * 100)}
    return {"n_rows": int(len(X)), "features": features}


def compute_feature_drift(
    profile: dict[str, Any], live: pd.DataFrame, *, min_samples: int = 30,
    psi_warning: float = 0.10, psi_alert: float = 0.25,
) -> dict[str, dict[str, Any]]:
    """Compare live data with the training reference profile (PSI + KS).

    Args:
        profile: Output of :func:`build_reference_profile`.
        live: Recent inference records (raw feature values).
        min_samples: Minimum non-null values required per feature.
        psi_warning: PSI at/above which status becomes ``warning``.
        psi_alert: PSI at/above which status becomes ``drift``.

    Returns:
        Mapping ``feature -> {"psi", "ks_pvalue", "status", "n"}``.
    """
    result: dict[str, dict[str, Any]] = {}
    for col, ref in profile.get("features", {}).items():
        if col not in live.columns:
            continue
        if ref["kind"] == "numeric":
            values = _coerce_numeric(live[col]).dropna().to_numpy(dtype=float)
            if len(values) < min_samples:
                result[col] = {"psi": None, "ks_pvalue": None, "status": "insufficient_data", "n": len(values)}
                continue
            inner = np.asarray(ref["inner_edges"], dtype=float)
            counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
            psi = population_stability_index(ref["expected"], counts / counts.sum())
            ks_p = float(stats.ks_2samp(np.asarray(ref["sample"]), values).pvalue)
        else:
            s = live[col].astype("object").where(live[col].notna(), MISSING_BUCKET).astype(str)
            if len(s) < min_samples:
                result[col] = {"psi": None, "ks_pvalue": None, "status": "insufficient_data", "n": len(s)}
                continue
            keys = list(ref["expected"].keys())
            mapped = s.where(s.isin(keys), OTHER_BUCKET)
            freq = mapped.value_counts(normalize=True)
            psi = population_stability_index(
                [ref["expected"][k] for k in keys], [float(freq.get(k, 0.0)) for k in keys]
            )
            ks_p = None
            values = s.to_numpy()
        status = "drift" if psi >= psi_alert else "warning" if psi >= psi_warning else "stable"
        result[col] = {"psi": round(psi, 5), "ks_pvalue": None if ks_p is None else round(ks_p, 6),
                       "status": status, "n": int(len(values))}
    return result


def build_prediction_profile(probabilities: np.ndarray, threshold: float, n_bins: int = 10) -> dict[str, Any]:
    """Baseline distribution of predicted churn probabilities (for shift analysis).

    Args:
        probabilities: Held-out predicted probabilities of the positive class.
        threshold: Decision threshold used to derive the expected churn rate.
        n_bins: Number of equal-width probability bins.

    Returns:
        Profile with expected positive rate, bin proportions and a KS sample.
    """
    probs = np.asarray(probabilities, dtype=float)
    counts, _ = np.histogram(probs, bins=np.linspace(0.0, 1.0, n_bins + 1))
    return {
        "threshold": float(threshold), "expected_positive_rate": float((probs >= threshold).mean()),
        "mean_probability": float(probs.mean()), "n_bins": n_bins,
        "expected": (counts / max(counts.sum(), 1)).tolist(),
        "sample": np.round(probs[:2000], 6).tolist(),
    }


def compute_prediction_shift(profile: dict[str, Any], live_probs: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    """Compare live prediction probabilities with the baseline prediction profile.

    Args:
        profile: Output of :func:`build_prediction_profile`.
        live_probs: Recently served probabilities.
        threshold: Threshold for the live positive rate (defaults to the baseline one).

    Returns:
        ``{"psi", "ks_pvalue", "live_positive_rate", "expected_positive_rate", "n"}``.
    """
    probs = np.asarray(live_probs, dtype=float)
    if len(probs) == 0:
        return {"psi": None, "ks_pvalue": None, "live_positive_rate": None,
                "expected_positive_rate": profile["expected_positive_rate"], "n": 0}
    thr = profile["threshold"] if threshold is None else threshold
    counts, _ = np.histogram(probs, bins=np.linspace(0.0, 1.0, profile["n_bins"] + 1))
    psi = population_stability_index(profile["expected"], counts / max(counts.sum(), 1))
    ks_p = float(stats.ks_2samp(np.asarray(profile["sample"]), probs).pvalue)
    return {"psi": round(psi, 5), "ks_pvalue": round(ks_p, 6),
            "live_positive_rate": float((probs >= thr).mean()),
            "expected_positive_rate": profile["expected_positive_rate"], "n": int(len(probs))}
