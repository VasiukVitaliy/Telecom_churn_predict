"""Leakage-free preprocessing built on ``sklearn.compose.ColumnTransformer``.

Every stateful step (imputers, scalers, encoders) lives *inside* the returned
transformer, so when it is the first stage of a pipeline it is fitted on
training folds only. Column typing comes from the dynamically inferred
:class:`~ml_core.data_validator.DatasetSchema`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml_core.data_validator import DatasetSchema

NumericImputation = Literal["median", "mean", "constant", "iterative"]
CategoricalImputation = Literal["most_frequent", "constant"]
NUMERIC_STRATEGIES: tuple[str, ...] = ("median", "mean", "constant", "iterative")
CATEGORICAL_STRATEGIES: tuple[str, ...] = ("most_frequent", "constant")
UNKNOWN_LABEL: str = "Unknown"


def coerce_numeric_frame(X: Any) -> pd.DataFrame:
    """Convert every column to float (blank / unparsable strings become ``NaN``).

    Stateless, therefore safe to apply before or inside cross-validation.

    Args:
        X: DataFrame or array-like of numeric-looking values.

    Returns:
        Float64 DataFrame.
    """
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    return frame.apply(pd.to_numeric, errors="coerce").astype("float64")


def _canonical(value: Any) -> Any:
    """Canonical string form so 0, 0.0 and '0' map to the same category."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def coerce_categorical_frame(X: Any) -> pd.DataFrame:
    """Convert every column to canonical strings, keeping ``NaN`` for missing values.

    Args:
        X: DataFrame or array-like of categorical values.

    Returns:
        Object-typed DataFrame.
    """
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    out = frame.astype("object").apply(lambda col: col.map(_canonical))
    return out.replace("", np.nan).astype("object")


def _numeric_imputer(strategy: NumericImputation, random_state: int) -> Any:
    """Build the numeric imputer for ``strategy``."""
    if strategy in ("median", "mean"):
        return SimpleImputer(strategy=strategy, keep_empty_features=True)
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    if strategy == "iterative":
        return IterativeImputer(
            initial_strategy="median", max_iter=10, random_state=random_state, keep_empty_features=True
        )
    raise ValueError(f"Unknown numeric imputation '{strategy}'. Choose from {NUMERIC_STRATEGIES}.")


def _categorical_imputer(strategy: CategoricalImputation) -> SimpleImputer:
    """Build the categorical imputer for ``strategy``."""
    if strategy == "most_frequent":
        return SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=UNKNOWN_LABEL, keep_empty_features=True)
    raise ValueError(f"Unknown categorical imputation '{strategy}'. Choose from {CATEGORICAL_STRATEGIES}.")


def build_preprocessor(
    schema: DatasetSchema,
    *,
    numeric_imputation: NumericImputation = "median",
    categorical_imputation: CategoricalImputation = "most_frequent",
    scale_numeric: bool = False,
    max_categories: int = 50,
    random_state: int = 42,
) -> ColumnTransformer:
    """Build an unfitted ``ColumnTransformer`` driven by the inferred schema.

    Numeric columns: coerce -> impute (median | mean | constant 0 | iterative) -> optional scaling.
    Categorical columns: canonicalise -> impute (mode | "Unknown") -> one-hot
    (unseen / rare categories are grouped or zero-encoded, never raising at inference).

    Args:
        schema: Schema whose ``numeric_columns`` / ``categorical_columns`` define the inputs.
        numeric_imputation: Numeric imputation strategy.
        categorical_imputation: Categorical imputation strategy.
        scale_numeric: Standardise numeric features (unnecessary for tree models).
        max_categories: Cap on one-hot levels per categorical column.
        random_state: Seed for the iterative imputer.

    Returns:
        Unfitted ``ColumnTransformer`` emitting dense float arrays.

    Raises:
        ValueError: If a strategy is unknown or the schema has no usable feature.

    Example:
        >>> from ml_core.data_validator import infer_schema
        >>> df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": ["x", "y", None, "x"]})
        >>> build_preprocessor(infer_schema(df)).fit_transform(df).shape
        (4, 3)
    """
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if schema.numeric_columns:
        steps: list[tuple[str, Any]] = [
            ("coerce", FunctionTransformer(coerce_numeric_frame, feature_names_out="one-to-one")),
            ("impute", _numeric_imputer(numeric_imputation, random_state)),
        ]
        if scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), schema.numeric_columns))
    if schema.categorical_columns:
        transformers.append((
            "cat",
            Pipeline([
                ("coerce", FunctionTransformer(coerce_categorical_frame, feature_names_out="one-to-one")),
                ("impute", _categorical_imputer(categorical_imputation)),
                ("onehot", OneHotEncoder(
                    handle_unknown="infrequent_if_exist", max_categories=max_categories,
                    sparse_output=False, dtype=np.float32,
                )),
            ]),
            schema.categorical_columns,
        ))
    if not transformers:
        raise ValueError("Schema contains no numeric or categorical feature columns.")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output feature names of a *fitted* preprocessor.

    Args:
        preprocessor: Fitted ``ColumnTransformer``.

    Returns:
        List of transformed feature names.
    """
    return [str(n) for n in preprocessor.get_feature_names_out()]
