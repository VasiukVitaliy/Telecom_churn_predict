"""Classifier factory, baseline hyper-parameters and Optuna search spaces."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import optuna
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ModelName = Literal["random_forest", "xgboost", "lightgbm"]
SUPPORTED_MODELS: tuple[str, ...] = ("random_forest", "xgboost", "lightgbm")

BASELINE_PARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 3, "max_features": "sqrt"},
    "xgboost": {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.85,
        "colsample_bytree": 0.85, "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 1.0,
    },
    "lightgbm": {
        "n_estimators": 300, "num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "subsample": 0.85,
        "colsample_bytree": 0.85, "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_samples": 20,
    },
}

# Default Optuna search spaces; users may override any entry (see ``suggest_params``).
# Spec: {"type": "int"|"float"|"categorical", "low", "high", "log", "choices"}
DEFAULT_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "random_forest": {
        "n_estimators": {"type": "int", "low": 100, "high": 600},
        "max_depth": {"type": "int", "low": 4, "high": 30},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5]},
    },
    "xgboost": {
        "n_estimators": {"type": "int", "low": 100, "high": 700},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 20.0, "log": True},
        "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "min_child_weight": {"type": "float", "low": 1.0, "high": 10.0},
    },
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 100, "high": 700},
        "num_leaves": {"type": "int", "low": 8, "high": 128},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 20.0, "log": True},
        "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "min_child_samples": {"type": "int", "low": 5, "high": 100},
    },
}


def _check_name(name: str) -> None:
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED_MODELS}.")


def build_classifier(
    name: str,
    params: Mapping[str, Any] | None = None,
    *,
    random_state: int = 42,
    class_weight_balanced: bool = False,
    scale_pos_weight: float | None = None,
    n_jobs: int = 1,
) -> ClassifierMixin:
    """Instantiate a classifier with baseline defaults overridden by ``params``.

    Args:
        name: One of ``random_forest``, ``xgboost``, ``lightgbm``.
        params: Hyper-parameters overriding the baseline defaults.
        random_state: Seed propagated to the estimator.
        class_weight_balanced: Enable cost-sensitive learning. RandomForest / LightGBM use
            ``class_weight="balanced"``; XGBoost uses ``scale_pos_weight``.
        scale_pos_weight: Negative/positive ratio for XGBoost (required when
            ``class_weight_balanced`` and ``name == "xgboost"``; ignored otherwise).
        n_jobs: Parallel threads for the estimator.

    Returns:
        Unfitted scikit-learn compatible classifier.

    Raises:
        ValueError: If ``name`` is not supported.

    Example:
        >>> build_classifier("lightgbm", {"n_estimators": 50}).n_estimators
        50
    """
    _check_name(name)
    merged: dict[str, Any] = {**BASELINE_PARAMS[name], **(dict(params) if params else {})}
    if name == "random_forest":
        if class_weight_balanced:
            merged["class_weight"] = "balanced_subsample"
        return RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **merged)
    if name == "xgboost":
        if class_weight_balanced and scale_pos_weight is not None:
            merged["scale_pos_weight"] = float(scale_pos_weight)
        return XGBClassifier(
            random_state=random_state, n_jobs=n_jobs, tree_method="hist",
            eval_metric="logloss", **merged,
        )
    if class_weight_balanced:
        merged["class_weight"] = "balanced"
    return LGBMClassifier(
        random_state=random_state, n_jobs=n_jobs, subsample_freq=1, verbose=-1, **merged
    )


def suggest_params(
    trial: optuna.Trial, name: str, overrides: Mapping[str, Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """Sample one hyper-parameter set for an Optuna trial.

    Args:
        trial: Active Optuna trial.
        name: Model name.
        overrides: Per-parameter spec overrides (user-defined grids), e.g.
            ``{"max_depth": {"type": "int", "low": 3, "high": 6}}``. Unknown
            parameter names are added as new dimensions.

    Returns:
        Parameter dictionary usable with :func:`build_classifier`.

    Raises:
        ValueError: If a spec has an unsupported ``type``.
    """
    _check_name(name)
    space = {**DEFAULT_SEARCH_SPACES[name], **{k: dict(v) for k, v in (overrides or {}).items()}}
    out: dict[str, Any] = {}
    for param, spec in space.items():
        kind = spec.get("type")
        if kind == "int":
            out[param] = trial.suggest_int(param, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False)))
        elif kind == "float":
            out[param] = trial.suggest_float(param, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        elif kind == "categorical":
            out[param] = trial.suggest_categorical(param, list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported search-space type '{kind}' for parameter '{param}'.")
    return out
