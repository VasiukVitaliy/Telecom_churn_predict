"""Dataset storage plus EDA and PCA computations exposed through the REST API."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ml_core.config import Settings
from ml_core.data_validator import (
    ColumnKind, DataValidationError, encode_target, find_target_column, infer_schema,
    load_dataset, load_dataset_bytes, normalize_frame,
)
from ml_core.preprocessor import build_preprocessor, get_feature_names

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,120}$")
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_ALLOWED_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq"}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a dataset id does not resolve to a stored file."""


class DatasetStore:
    """Read / write access to ``data/raw`` and ``data/synthetic`` (path-traversal safe).

    Args:
        settings: Settings supplying the two data directories.
    """

    def __init__(self, settings: Settings) -> None:
        self._dirs: dict[str, Path] = {"raw": settings.data_raw_dir, "synthetic": settings.data_synthetic_dir}
        for directory in self._dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    def resolve(self, dataset_id: str) -> Path:
        """Map ``"raw/file.csv"`` to an absolute path.

        Raises:
            DatasetNotFoundError: If the id is malformed or the file does not exist.
        """
        source, _, name = dataset_id.partition("/")
        if source not in self._dirs or not _SAFE_NAME.match(name) or Path(name).suffix.lower() not in _ALLOWED_SUFFIXES:
            raise DatasetNotFoundError(f"Invalid dataset id '{dataset_id}'.")
        path = self._dirs[source] / name
        if not path.is_file():
            raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found.")
        return path

    def load(self, dataset_id: str) -> pd.DataFrame:
        """Load a stored dataset."""
        return load_dataset(self.resolve(dataset_id))

    def list(self) -> list[dict[str, Any]]:
        """List all stored datasets."""
        items: list[dict[str, Any]] = []
        for source, directory in self._dirs.items():
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in _ALLOWED_SUFFIXES:
                    items.append({"dataset_id": f"{source}/{path.name}", "source": source,
                                  "filename": path.name, "size_bytes": path.stat().st_size})
        return items

    def save_upload(self, filename: str, content: bytes) -> tuple[str, pd.DataFrame]:
        """Validate and store an uploaded file in ``data/raw``.

        Args:
            filename: Client-supplied file name.
            content: File bytes.

        Returns:
            ``(dataset_id, dataframe)``.

        Raises:
            DataValidationError: If the name / size / content is invalid.
        """
        name = Path(filename).name
        if not _SAFE_NAME.match(name) or Path(name).suffix.lower() not in _ALLOWED_SUFFIXES:
            raise DataValidationError("File name must match [A-Za-z0-9_.-] and end with .csv/.tsv/.parquet.")
        if len(content) > _MAX_UPLOAD_BYTES:
            raise DataValidationError("File exceeds the 200 MB upload limit.")
        df = load_dataset_bytes(content, name)
        (self._dirs["raw"] / name).write_bytes(content)
        return f"raw/{name}", df

    def save_frame(self, df: pd.DataFrame, filename: str, source: str = "synthetic") -> str:
        """Store a dataframe as CSV / Parquet."""
        path = self._dirs[source] / filename
        df.to_parquet(path, index=False) if path.suffix == ".parquet" else df.to_csv(path, index=False)
        return f"{source}/{filename}"


def _target_or_none(frame: pd.DataFrame, target: str | None) -> str | None:
    try:
        return find_target_column(frame, target)
    except DataValidationError:
        return None


def eda_summary(df: pd.DataFrame, target: str | None, settings: Settings) -> dict[str, Any]:
    """Dataset-level EDA: schema, missingness, target balance, correlations.

    Args:
        df: Dataset.
        target: Optional target column name.
        settings: Settings (cardinality cap).

    Returns:
        JSON-friendly summary dictionary.
    """
    frame = normalize_frame(df)
    tgt = _target_or_none(frame, target)
    schema = infer_schema(frame, tgt, max_cardinality=settings.max_categorical_cardinality)
    y = encode_target(frame[tgt]) if tgt else None
    numeric = pd.DataFrame({c: pd.to_numeric(frame[c], errors="coerce") for c in schema.numeric_columns})
    corr_cols = numeric.var().sort_values(ascending=False).index[:25].tolist() if not numeric.empty else []
    corr_frame = numeric[corr_cols].copy()
    if y is not None and corr_cols:
        corr_frame["__target__"] = y
    corr = corr_frame.corr() if corr_cols else pd.DataFrame()
    desc = numeric.describe().T if not numeric.empty else pd.DataFrame()
    return {
        "n_rows": len(frame), "n_columns": frame.shape[1], "target": tgt,
        "positive_rate": None if y is None else float(y.mean()),
        "target_counts": None if y is None else {str(int(k)): int(v) for k, v in y.value_counts().items()},
        "columns": [{k: v for k, v in c.to_dict().items() if k != "categories"} for c in schema.columns.values()],
        "missing_pct": {c: round(float(frame[c].isna().mean() * 100), 3) for c in frame.columns
                        if frame[c].isna().any()},
        "numeric_summary": desc.round(4).to_dict("index"),
        "correlation": {"labels": corr.columns.tolist(), "matrix": corr.round(4).values.tolist()},
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def column_profile(df: pd.DataFrame, column: str, target: str | None, bins: int = 30) -> dict[str, Any]:
    """Distribution of one column, split by the target class.

    Args:
        df: Dataset.
        column: Column to profile.
        target: Optional target column name.
        bins: Histogram bin count (numeric columns).

    Returns:
        Histogram (numeric) or value counts with churn rate (categorical).

    Raises:
        DataValidationError: If the column does not exist.
    """
    frame = normalize_frame(df)
    if column not in frame.columns:
        raise DataValidationError(f"Column '{column}' not found.")
    tgt = _target_or_none(frame, target)
    y = encode_target(frame[tgt]) if tgt else None
    schema = infer_schema(frame, tgt)
    spec = schema.columns[column]
    if spec.kind == ColumnKind.NUMERIC:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        edges = np.histogram_bin_edges(valid, bins=bins) if len(valid) else np.array([0.0, 1.0])
        out: dict[str, Any] = {"kind": "numeric", "edges": edges.tolist(),
                               "counts": np.histogram(valid, bins=edges)[0].tolist(),
                               "stats": valid.describe().to_dict(), "missing": int(values.isna().sum())}
        if y is not None:
            out["counts_by_class"] = {
                str(int(c)): np.histogram(values[(y == c)].dropna(), bins=edges)[0].tolist() for c in (0, 1)
            }
        return out
    counts = frame[column].astype("object").fillna("(missing)").astype(str).value_counts().head(30)
    out = {"kind": spec.kind.value, "categories": counts.index.tolist(), "counts": counts.tolist(),
           "missing": int(frame[column].isna().sum())}
    if y is not None:
        key = frame[column].astype("object").fillna("(missing)").astype(str)
        rate = y.groupby(key).mean()
        out["churn_rate"] = [float(rate.get(c, np.nan)) for c in counts.index]
    return out


def pca_projection(
    df: pd.DataFrame, target: str | None, settings: Settings, *, n_components: int = 3,
    sample_size: int = 3000,
) -> dict[str, Any]:
    """PCA of the (imputed, scaled, one-hot encoded) feature space.

    Args:
        df: Dataset.
        target: Optional target column name.
        settings: Settings (seed, cardinality cap).
        n_components: Components to compute (2-10).
        sample_size: Maximum rows projected (random sample).

    Returns:
        Explained variance, projected points with labels, and top loadings for PC1/PC2.

    Raises:
        DataValidationError: If the dataset has too few usable rows or features.
    """
    frame = normalize_frame(df)
    tgt = _target_or_none(frame, target)
    schema = infer_schema(frame, tgt, max_cardinality=settings.max_categorical_cardinality)
    if len(frame) < 10 or not schema.feature_columns:
        raise DataValidationError("Dataset too small or without usable feature columns for PCA.")
    if len(frame) > sample_size:
        frame = frame.sample(sample_size, random_state=settings.random_state)
    pre = build_preprocessor(schema, scale_numeric=True, max_categories=settings.max_categorical_cardinality,
                             random_state=settings.random_state)
    matrix = pre.fit_transform(frame[schema.feature_columns])
    k = int(max(2, min(n_components, 10, matrix.shape[1], matrix.shape[0] - 1)))
    pca = PCA(n_components=k, random_state=settings.random_state)
    coords = pca.fit_transform(matrix)
    names = get_feature_names(pre)
    loadings: dict[str, list[dict[str, Any]]] = {}
    for i in range(min(2, k)):
        order = np.argsort(np.abs(pca.components_[i]))[::-1][:10]
        loadings[f"PC{i + 1}"] = [{"feature": names[j], "loading": float(pca.components_[i][j])} for j in order]
    labels = encode_target(frame[tgt]).tolist() if tgt else [None] * len(frame)
    return {
        "n_components": k, "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "points": {f"PC{i + 1}": coords[:, i].round(5).tolist() for i in range(k)},
        "labels": labels, "loadings": loadings, "n_points": int(len(frame)),
    }
