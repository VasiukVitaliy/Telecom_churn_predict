"""Unit tests for data loading, schema inference, quality gates, drift maths and the generator."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from ml_core.config import Settings, resolve_db_uri, resolve_path
from ml_core.data_generator import build_sampler, generate_synthetic_subscribers, inject_missing_values
from ml_core.data_validator import (
    ColumnKind, DataValidationError, DatasetSchema, build_prediction_profile, build_reference_profile,
    compute_feature_drift, compute_prediction_shift, encode_target, find_target_column, infer_schema,
    load_dataset_bytes, normalize_frame, population_stability_index, prepare_training_frame,
    validate_dataframe,
)


def telco_like(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Telco-style frame incl. blank-string numerics and an id column."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    monthly = np.round(rng.uniform(20, 110, n), 2)
    total = (monthly * tenure).round(2).astype(str)
    total[tenure == 0] = " "  # the classic Telco quirk
    churn = np.where(rng.random(n) < 0.25, "Yes", "No")
    return pd.DataFrame({
        "customerID": [f"C-{i:05d}" for i in range(n)], "tenure": tenure, "MonthlyCharges": monthly,
        "TotalCharges": total, "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "Churn": churn,
    })


# ------------------------------------------------------------------ loading
def test_load_csv_bytes_strips_column_names() -> None:
    df = load_dataset_bytes(b" a , b \n1,2\n3,4\n", "x.csv")
    assert list(df.columns) == ["a", "b"]


def test_load_parquet_roundtrip() -> None:
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(buf)
    assert len(load_dataset_bytes(buf.getvalue(), "x.parquet")) == 3


@pytest.mark.parametrize("content,name", [(b"a,b\n", "empty.csv"), (b"x", "file.xlsx")])
def test_load_rejects_empty_and_unsupported(content: bytes, name: str) -> None:
    with pytest.raises(DataValidationError):
        load_dataset_bytes(content, name)


# --------------------------------------------------------------- schema
def test_infer_schema_detects_roles() -> None:
    schema = infer_schema(telco_like(), "Churn")
    assert schema.columns["customerID"].kind == ColumnKind.ID
    assert schema.columns["Churn"].kind == ColumnKind.TARGET
    assert schema.columns["Contract"].kind == ColumnKind.CATEGORICAL
    assert {"tenure", "MonthlyCharges"} <= set(schema.numeric_columns)
    assert "customerID" not in schema.feature_columns


def test_blank_string_numeric_is_typed_numeric() -> None:
    schema = infer_schema(telco_like(), "Churn")
    assert schema.columns["TotalCharges"].kind == ColumnKind.NUMERIC


def test_schema_dict_roundtrip() -> None:
    schema = infer_schema(telco_like(), "Churn")
    assert DatasetSchema.from_dict(schema.to_dict()).feature_columns == schema.feature_columns


def test_constant_column_is_not_a_feature() -> None:
    df = telco_like()
    df["const"] = 1
    assert "const" not in infer_schema(df, "Churn").feature_columns


# --------------------------------------------------------------- target
def test_find_target_is_case_insensitive_and_explicit_wins() -> None:
    assert find_target_column(telco_like()) == "Churn"
    assert find_target_column(telco_like(), "churn") == "Churn"
    with pytest.raises(DataValidationError):
        find_target_column(telco_like(), "nope")


def test_encode_target_labels() -> None:
    out = encode_target(pd.Series(["Yes", "No", "true", "0", None]))
    assert out.iloc[:4].tolist() == [1.0, 0.0, 1.0, 0.0] and np.isnan(out.iloc[4])
    with pytest.raises(DataValidationError):
        encode_target(pd.Series(["maybe"]))


# ------------------------------------------------------- quality gates
def test_missing_threshold_rejects_training_data(tmp_settings: Settings) -> None:
    df = telco_like()
    df.loc[df.sample(frac=0.45, random_state=1).index, "MonthlyCharges"] = np.nan
    with pytest.raises(DataValidationError, match="MonthlyCharges"):
        validate_dataframe(df, mode="training", settings=tmp_settings)


def test_missing_threshold_is_configurable(tmp_path) -> None:
    from tests.conftest import make_settings

    df = telco_like()
    df.loc[df.sample(frac=0.45, random_state=1).index, "MonthlyCharges"] = np.nan
    lenient = make_settings(tmp_path, max_missing_fraction=0.6)
    assert validate_dataframe(df, mode="training", settings=lenient).passed


def test_negative_tenure_and_charges_fail(tmp_settings: Settings) -> None:
    df = telco_like()
    df.loc[0, "tenure"] = -3
    df.loc[1, "MonthlyCharges"] = -10.0
    report = validate_dataframe(df, mode="training", settings=tmp_settings, raise_on_error=False)
    assert not report.passed
    assert any("tenure" in e and "negative" in e for e in report.errors)
    assert any("MonthlyCharges" in e for e in report.errors)


def test_signed_columns_may_be_negative(tmp_settings: Settings) -> None:
    df = telco_like()
    df["avg_bill_change_pct"] = np.random.default_rng(3).normal(0, 5, len(df))
    assert validate_dataframe(df, mode="training", settings=tmp_settings).passed


def test_too_few_rows_and_single_class(tmp_settings: Settings) -> None:
    small = telco_like(50)
    assert not validate_dataframe(small, mode="training", settings=tmp_settings, raise_on_error=False).passed
    one_class = telco_like()
    one_class["Churn"] = "No"
    report = validate_dataframe(one_class, mode="training", settings=tmp_settings, raise_on_error=False)
    assert any("single class" in e.lower() for e in report.errors)


def test_inference_schema_similarity(tmp_settings: Settings) -> None:
    train = telco_like()
    schema = infer_schema(train, "Churn")
    live = train.drop(columns=["Churn"]).head(50).copy()
    assert validate_dataframe(live, schema, mode="inference", settings=tmp_settings).passed

    missing = live.drop(columns=["tenure"])
    report = validate_dataframe(missing, schema, mode="inference", settings=tmp_settings, raise_on_error=False)
    assert any("Missing required columns" in e for e in report.errors)

    extra = live.assign(surprise=1)
    lenient = validate_dataframe(extra, schema, mode="inference", settings=tmp_settings)
    assert any("Unexpected columns" in w for w in lenient.warnings)
    strict = validate_dataframe(extra, schema, mode="inference", settings=tmp_settings,
                                strict_extra_columns=True, raise_on_error=False)
    assert not strict.passed

    bad_type = live.copy()
    bad_type["MonthlyCharges"] = bad_type["MonthlyCharges"].astype(object)
    bad_type.loc[bad_type.index[0], "MonthlyCharges"] = "abc"
    assert not validate_dataframe(bad_type, schema, mode="inference", settings=tmp_settings,
                                  raise_on_error=False).passed


def test_inference_requires_schema(tmp_settings: Settings) -> None:
    with pytest.raises(DataValidationError):
        validate_dataframe(telco_like(), None, mode="inference", settings=tmp_settings)


def test_prepare_training_frame_splits_target(tmp_settings: Settings) -> None:
    prepared = prepare_training_frame(telco_like(), settings=tmp_settings)
    assert "Churn" not in prepared.X.columns and "customerID" not in prepared.X.columns
    assert set(prepared.y.unique()) == {0, 1}
    assert len(prepared.X) == len(prepared.y)


def test_normalize_frame_turns_blanks_into_nan() -> None:
    out = normalize_frame(pd.DataFrame({"a": ["x", " ", ""]}))
    assert out["a"].isna().tolist() == [False, True, True]


# --------------------------------------------------------------- drift
def test_psi_zero_for_identical_and_large_for_shifted() -> None:
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert population_stability_index(p, p) == pytest.approx(0.0, abs=1e-9)
    assert population_stability_index(p, [0.7, 0.1, 0.1, 0.1]) > 0.25


def test_feature_drift_detects_shift(tmp_settings: Settings) -> None:
    df = telco_like(2000)
    schema = infer_schema(df, "Churn")
    profile = build_reference_profile(df[schema.feature_columns], schema)
    same = compute_feature_drift(profile, df.sample(500, random_state=1)[schema.feature_columns])
    assert same["MonthlyCharges"]["status"] == "stable"
    shifted = df.sample(500, random_state=1)[schema.feature_columns].copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 2 + 50
    shifted["Contract"] = "Month-to-month"
    drift = compute_feature_drift(profile, shifted)
    assert drift["MonthlyCharges"]["status"] == "drift"
    assert drift["Contract"]["status"] == "drift"


def test_feature_drift_insufficient_data() -> None:
    df = telco_like(300)
    schema = infer_schema(df, "Churn")
    profile = build_reference_profile(df[schema.feature_columns], schema)
    res = compute_feature_drift(profile, df.head(5)[schema.feature_columns], min_samples=30)
    assert all(v["status"] == "insufficient_data" for v in res.values())


def test_prediction_shift() -> None:
    rng = np.random.default_rng(0)
    base = rng.beta(2, 8, 3000)
    profile = build_prediction_profile(base, 0.5)
    stable = compute_prediction_shift(profile, rng.beta(2, 8, 800))
    moved = compute_prediction_shift(profile, rng.beta(6, 3, 800))
    assert stable["psi"] < 0.1 < 0.25 < moved["psi"]
    assert compute_prediction_shift(profile, np.array([]))["n"] == 0


# ----------------------------------------------------------- generator
def test_generator_shape_rate_and_determinism() -> None:
    a = generate_synthetic_subscribers(2000, 0.2, 7)
    b = generate_synthetic_subscribers(2000, 0.2, 7)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape[1] >= 30 and len(a) == 2000
    assert 0.15 <= a["churn"].mean() <= 0.25


def test_generator_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_subscribers(10)
    with pytest.raises(ValueError):
        generate_synthetic_subscribers(1000, 0.9)


def test_inject_missing_values_adds_nans() -> None:
    df = generate_synthetic_subscribers(1000, 0.2, 1)
    out = inject_missing_values(df, 0.1, random_state=1)
    assert out.isna().sum().sum() > df.isna().sum().sum()


def test_sampler_factory() -> None:
    assert build_sampler("none") is None and build_sampler("class_weight") is None
    assert build_sampler("smote", minority_ratio=0.2) is not None
    assert build_sampler("random_over", minority_ratio=0.7, sampling_ratio=0.6) is None
    with pytest.raises(ValueError):
        build_sampler("bogus")  # type: ignore[arg-type]


# -------------------------------------------------------------- config
def test_path_resolution_is_cwd_independent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_path("data/raw").is_absolute()
    assert str(tmp_path) not in str(resolve_path("data/raw"))
    uri = resolve_db_uri("sqlite:///models/x.db")
    assert uri.startswith("sqlite:////") and str(tmp_path) not in uri
    assert resolve_db_uri("postgresql+psycopg2://u:p@h/db") == "postgresql+psycopg2://u:p@h/db"
