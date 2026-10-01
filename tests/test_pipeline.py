"""Tests for the leakage-free preprocessor, imputation strategies, model factory and trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer

from ml_core.config import Settings
from ml_core.data_validator import infer_schema, prepare_training_frame
from ml_core.models import BASELINE_PARAMS, SUPPORTED_MODELS, build_classifier, suggest_params
from ml_core.preprocessor import (
    build_preprocessor, coerce_categorical_frame, coerce_numeric_frame, get_feature_names,
)
from ml_core.registry import ModelRegistry
from ml_core.trainer import (
    ModelTrainer, TrainingConfig, TrainingJobManager, build_pipeline, compute_metrics,
    cross_validate_pipeline,
)


def small_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        "b": ["x", "y", None, "x", "x", "y"],
        "c": [10.0, 20.0, 30.0, np.nan, 50.0, 60.0],
    })


def frame_schema(df: pd.DataFrame):
    return infer_schema(df)


# ------------------------------------------------------------ coercion
def test_coerce_numeric_turns_garbage_into_nan() -> None:
    out = coerce_numeric_frame(pd.DataFrame({"a": ["1", " ", "x", 2.5]}))
    assert out["a"].isna().tolist() == [False, True, True, False]
    assert str(out["a"].dtype) == "float64"


def test_categorical_canonicalisation_unifies_representations() -> None:
    out = coerce_categorical_frame(pd.DataFrame({"a": [0, 0.0, "0", " 0 ", None, ""]}))
    assert out["a"].iloc[:4].tolist() == ["0"] * 4
    assert out["a"].iloc[4:].isna().all()


# ---------------------------------------------------- imputation strategies
@pytest.mark.parametrize("strategy,expected", [("median", 4.0), ("mean", 3.8), ("constant", 0.0)])
def test_numeric_imputation_values(strategy: str, expected: float) -> None:
    df = small_frame()
    pre = build_preprocessor(frame_schema(df), numeric_imputation=strategy).fit(df)  # type: ignore[arg-type]
    names = get_feature_names(pre)
    out = pre.transform(df)
    assert out[1, names.index("a")] == pytest.approx(expected)


def test_iterative_imputation_fills_everything() -> None:
    df = small_frame()
    out = build_preprocessor(frame_schema(df), numeric_imputation="iterative").fit_transform(df)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("strategy,expected_col", [("most_frequent", "b_x"), ("constant", "b_Unknown")])
def test_categorical_imputation(strategy: str, expected_col: str) -> None:
    df = small_frame()
    pre = build_preprocessor(frame_schema(df), categorical_imputation=strategy).fit(df)  # type: ignore[arg-type]
    names = get_feature_names(pre)
    row = pre.transform(df)[2]  # the row whose category was missing
    assert expected_col in names and row[names.index(expected_col)] == 1.0


def test_unknown_strategy_and_empty_schema_raise() -> None:
    df = small_frame()
    with pytest.raises(ValueError):
        build_preprocessor(frame_schema(df), numeric_imputation="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_preprocessor(infer_schema(pd.DataFrame({"only": [1, 1, 1]})))


def test_preprocessor_is_a_column_transformer_and_handles_unseen_categories() -> None:
    df = small_frame()
    pre = build_preprocessor(frame_schema(df))
    assert isinstance(pre, ColumnTransformer)
    pre.fit(df)
    unseen = pd.DataFrame({"a": [1.0], "b": ["NEVER_SEEN"], "c": [5.0]})
    assert pre.transform(unseen).shape[0] == 1  # must not raise at inference


def test_scaling_is_optional() -> None:
    df = small_frame()
    scaled = build_preprocessor(frame_schema(df), scale_numeric=True).fit_transform(df)
    assert abs(scaled[:, 0].mean()) < 1e-6


# ------------------------------------------------------- leakage freedom
def test_imputer_statistics_come_from_training_data_only() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": ["x", "y", "x", "y"]})
    valid = pd.DataFrame({"a": [1000.0, np.nan], "b": ["x", "y"]})
    pre = build_preprocessor(frame_schema(train), numeric_imputation="median").fit(train)
    out = pre.transform(valid)
    names = get_feature_names(pre)
    assert out[1, names.index("a")] == pytest.approx(2.0)  # train median, not influenced by 1000.0


def test_resampling_only_happens_while_fitting(synthetic_df: pd.DataFrame, tmp_settings: Settings) -> None:
    prepared = prepare_training_frame(synthetic_df, settings=tmp_settings)
    cfg = TrainingConfig(model_name="lightgbm", sampling="random_over", sampling_ratio=0.9, **{"cv_folds": 2})
    pipe = build_pipeline(prepared.schema, cfg, {"n_estimators": 20}, minority_ratio=0.25, scale_pos_weight=4.0,
                          settings=tmp_settings, n_jobs=1)
    assert isinstance(pipe, ImbPipeline) and [n for n, _ in pipe.steps] == ["preprocess", "sampler", "clf"]
    X, y = prepared.X.head(600), prepared.y.head(600)
    pipe.fit(X, y)
    # predictions keep the exact number of input rows -> the sampler is skipped at predict time
    assert pipe.predict_proba(X.head(37)).shape == (37, 2)


@pytest.mark.parametrize("sampling,steps", [
    ("none", ["preprocess", "clf"]), ("class_weight", ["preprocess", "clf"]),
    ("smote", ["preprocess", "sampler", "clf"]),
])
def test_pipeline_structure(sampling: str, steps: list[str], synthetic_df: pd.DataFrame,
                            tmp_settings: Settings) -> None:
    prepared = prepare_training_frame(synthetic_df, settings=tmp_settings)
    cfg = TrainingConfig(sampling=sampling, sampling_ratio=0.9)  # type: ignore[arg-type]
    pipe = build_pipeline(prepared.schema, cfg, None, minority_ratio=0.25, scale_pos_weight=3.0,
                          settings=tmp_settings, n_jobs=1)
    assert [n for n, _ in pipe.steps] == steps


# ------------------------------------------------------------ model factory
@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_factory_builds_each_model_and_overrides_params(name: str) -> None:
    clf = build_classifier(name, {"n_estimators": 11}, random_state=7)
    assert clf.get_params()["n_estimators"] == 11 and clf.get_params()["random_state"] == 7
    assert set(BASELINE_PARAMS[name]) - {"n_estimators"} <= set(clf.get_params())


def test_factory_class_weight_handling() -> None:
    assert build_classifier("random_forest", class_weight_balanced=True).class_weight == "balanced_subsample"
    assert build_classifier("lightgbm", class_weight_balanced=True).class_weight == "balanced"
    assert build_classifier("xgboost", class_weight_balanced=True, scale_pos_weight=3.5).scale_pos_weight == 3.5
    with pytest.raises(ValueError):
        build_classifier("svm")


def test_suggest_params_respects_overrides() -> None:
    import optuna

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    trial = study.ask()
    params = suggest_params(trial, "xgboost", {"max_depth": {"type": "int", "low": 3, "high": 4},
                                               "extra": {"type": "categorical", "choices": ["a"]}})
    assert params["max_depth"] in (3, 4) and params["extra"] == "a"
    with pytest.raises(ValueError):
        suggest_params(study.ask(), "lightgbm", {"x": {"type": "weird"}})


# ---------------------------------------------------------------- metrics
def test_compute_metrics_keys_and_values() -> None:
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.4, 0.6, 0.9]))
    assert set(m) == {"roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "brier_score"}
    assert m["roc_auc"] == 1.0 and m["accuracy"] == 1.0
    with pytest.raises(ValueError):
        compute_metrics(np.array([1, 1]), np.array([0.2, 0.9]))


def test_cross_validation_refits_per_fold(synthetic_df: pd.DataFrame, tmp_settings: Settings) -> None:
    prepared = prepare_training_frame(synthetic_df, settings=tmp_settings)
    cfg = TrainingConfig(model_name="lightgbm")
    pipe = build_pipeline(prepared.schema, cfg, {"n_estimators": 30}, minority_ratio=0.25, scale_pos_weight=4.0,
                          settings=tmp_settings, n_jobs=1)
    cv = cross_validate_pipeline(pipe, prepared.X.head(800), prepared.y.head(800), n_splits=3, random_state=42)
    assert 0.5 < cv["roc_auc"]["mean"] <= 1.0 and cv["roc_auc"]["std"] >= 0.0
    assert not hasattr(pipe, "classes_")  # the template pipeline itself was never fitted


# ------------------------------------------------------------ training engine
def test_baseline_training_registers_a_version(synthetic_df: pd.DataFrame, tmp_settings: Settings) -> None:
    registry = ModelRegistry(tmp_settings)
    trainer = ModelTrainer(registry, tmp_settings)
    cfg = TrainingConfig.from_settings(tmp_settings, model_name="lightgbm", cv_folds=2)
    result = trainer.train(synthetic_df, cfg)
    assert result.model_version == 1
    assert result.test_metrics["roc_auc"] > 0.6
    assert len(result.confusion_matrix) == 2
    assert registry.list_model_versions()[0]["version"] == 1


def test_training_is_deterministic(synthetic_df: pd.DataFrame, tmp_settings: Settings) -> None:
    trainer = ModelTrainer(ModelRegistry(tmp_settings), tmp_settings)
    cfg = TrainingConfig.from_settings(tmp_settings, model_name="random_forest", cv_folds=2, register=False)
    a = trainer.train(synthetic_df.head(800), cfg).test_metrics
    b = trainer.train(synthetic_df.head(800), cfg).test_metrics
    assert a == pytest.approx(b)


def test_optuna_mode_returns_best_params(synthetic_df: pd.DataFrame, tmp_settings: Settings) -> None:
    trainer = ModelTrainer(ModelRegistry(tmp_settings), tmp_settings)
    cfg = TrainingConfig.from_settings(
        tmp_settings, model_name="lightgbm", mode="optuna", n_trials=3, cv_folds=2, register=False,
        optimize_metric="roc_auc",
        param_space={"n_estimators": {"type": "int", "low": 20, "high": 40},
                     "num_leaves": {"type": "int", "low": 8, "high": 16}},
    )
    result = trainer.train(synthetic_df.head(900), cfg)
    assert result.n_trials == 3
    assert 20 <= result.best_params["n_estimators"] <= 40 and result.model_version is None


def test_job_manager_records_success_and_failure(synthetic_df: pd.DataFrame, tmp_settings: Settings) -> None:
    jobs = TrainingJobManager(ModelTrainer(ModelRegistry(tmp_settings), tmp_settings))
    ok_cfg = TrainingConfig.from_settings(tmp_settings, model_name="lightgbm", cv_folds=2, register=False)
    ok = jobs.create_job(ok_cfg)
    jobs.run_job(ok, synthetic_df.head(700), ok_cfg)
    assert jobs.get_job(ok)["status"] == "succeeded"

    bad = jobs.create_job(ok_cfg)
    jobs.run_job(bad, synthetic_df.head(700).drop(columns=["churn"]), ok_cfg)
    record = jobs.get_job(bad)
    assert record["status"] == "failed" and "target" in record["error"].lower()
    assert jobs.get_job("nope") is None and len(jobs.list_jobs()) == 2
    jobs.shutdown()


def test_training_config_validation() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(model_name="svm")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TrainingConfig(unknown_field=1)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        TrainingConfig(test_size=0.9)
