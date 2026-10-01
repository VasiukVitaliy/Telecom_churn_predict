"""FastAPI endpoint / contract tests (TestClient, isolated storage, tiny trained models)."""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.conftest import feature_record

pytestmark = pytest.mark.usefixtures("trained_client")


# ------------------------------------------------------------------ health
def test_health_reports_loaded_model(trained_client: TestClient) -> None:
    body = trained_client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["model_version"] == 1
    assert body["tracking_uri"].startswith("sqlite:///") and "X-Process-Time-Ms" in trained_client.get("/health").headers


def test_degraded_without_model(empty_client: TestClient) -> None:
    body = empty_client.get("/health").json()
    assert body["status"] == "degraded" and body["model_loaded"] is False


def test_inference_without_model_returns_503(empty_client: TestClient) -> None:
    assert empty_client.post("/predict", json={"features": {"a": 1}}).status_code == 503
    assert empty_client.get("/model/info").status_code == 503
    assert empty_client.post("/model/reload").status_code == 404  # no @champion alias yet


# ---------------------------------------------------------------- datasets
def test_dataset_listing_and_synthetic_generation(trained_client: TestClient) -> None:
    ids = [d["dataset_id"] for d in trained_client.get("/datasets").json()]
    assert "synthetic/ci.csv" in ids
    made = trained_client.post("/datasets/synthetic", json={"n_samples": 600, "random_state": 3, "filename": "tiny.parquet"})
    assert made.status_code == 201 and made.json()["n_rows"] == 600


@pytest.mark.parametrize("payload", [{"n_samples": 10}, {"target_churn_rate": 0.9}, {"filename": "../evil.csv"},
                                     {"surprise": 1}])
def test_synthetic_request_validation(trained_client: TestClient, payload: dict[str, Any]) -> None:
    assert trained_client.post("/datasets/synthetic", json=payload).status_code == 422


def test_upload_real_dataset_and_reject_bad_files(trained_client: TestClient) -> None:
    csv = pd.DataFrame({"x": range(300), "y": ["a", "b", "c"] * 100, "churn": ["Yes", "No", "No"] * 100}).to_csv(index=False)
    ok = trained_client.post("/datasets/upload", files={"file": ("real.csv", csv.encode())})
    assert ok.status_code == 201 and ok.json()["dataset_id"] == "raw/real.csv" and ok.json()["n_rows"] == 300
    bad_name = trained_client.post("/datasets/upload", files={"file": ("bad name!.csv", csv.encode())})
    assert bad_name.status_code == 422
    bad_type = trained_client.post("/datasets/upload", files={"file": ("x.exe", b"MZ")})
    assert bad_type.status_code == 422


def test_path_traversal_is_blocked(trained_client: TestClient) -> None:
    assert trained_client.get("/datasets/raw/..%2F..%2Fetc%2Fpasswd/eda").status_code in (404, 422)
    assert trained_client.post("/datasets/validate", json={"dataset_id": "raw/../x.csv"}).status_code == 404


def test_validate_endpoint_returns_schema_and_report(trained_client: TestClient) -> None:
    body = trained_client.post("/datasets/validate", json={"dataset_id": "synthetic/ci.csv"}).json()
    assert body["target"] == "churn" and body["report"]["passed"] is True
    assert body["schema"]["columns"]["subscriber_id"]["kind"] == "id"
    assert len(body["feature_columns"]) >= 30


def test_validate_reports_failures_without_raising(trained_client: TestClient) -> None:
    df = pd.DataFrame({"usage": [float(i % 17) for i in range(300)],
                       "tenure": [-(i % 5) - 1 if i < 150 else i % 40 for i in range(300)],  # many distinct values
                       "churn": ["Yes", "No"] * 150})
    trained_client.post("/datasets/upload", files={"file": ("neg.csv", df.to_csv(index=False).encode())})
    body = trained_client.post("/datasets/validate", json={"dataset_id": "raw/neg.csv"}).json()
    assert body["report"]["passed"] is False and any("tenure" in e for e in body["report"]["errors"])


def test_eda_column_and_pca(trained_client: TestClient) -> None:
    eda = trained_client.get("/datasets/synthetic/ci.csv/eda").json()
    assert eda["n_rows"] == 1200 and 0.1 < eda["positive_rate"] < 0.35 and eda["correlation"]["labels"]
    num = trained_client.get("/datasets/synthetic/ci.csv/columns/monthly_charges").json()
    assert num["kind"] == "numeric" and set(num["counts_by_class"]) == {"0", "1"}
    cat = trained_client.get("/datasets/synthetic/ci.csv/columns/contract_type").json()
    assert cat["categories"] and len(cat["churn_rate"]) == len(cat["categories"])
    assert trained_client.get("/datasets/synthetic/ci.csv/columns/nope").status_code == 422
    pca = trained_client.get("/datasets/synthetic/ci.csv/pca", params={"n_components": 3, "sample_size": 500}).json()
    assert pca["n_components"] == 3 and len(pca["points"]["PC1"]) == pca["n_points"] == 500
    assert sum(pca["explained_variance_ratio"]) <= 1.0 + 1e-9


def test_unknown_dataset_is_404(trained_client: TestClient) -> None:
    assert trained_client.get("/datasets/raw/missing.csv/eda").status_code == 404


# ---------------------------------------------------------------- training
def test_training_job_lifecycle_and_failure(trained_client: TestClient) -> None:
    jobs = trained_client.get("/training/jobs").json()
    assert len(jobs) >= 2 and all(j["status"] == "succeeded" for j in jobs)
    job = trained_client.get(f"/training/jobs/{jobs[0]['job_id']}").json()
    assert job["result"]["test_metrics"]["roc_auc"] > 0.6
    assert trained_client.get("/training/jobs/ghost").status_code == 404


def test_training_rejects_bad_config_and_bad_data(trained_client: TestClient) -> None:
    r = trained_client.post("/training/start", json={"dataset_id": "synthetic/ci.csv", "config": {"model_name": "svm"}})
    assert r.status_code == 422
    r = trained_client.post("/training/start", json={"dataset_id": "synthetic/ci.csv", "config": {"cv_folds": 1}})
    assert r.status_code == 422
    r = trained_client.post("/training/start", json={"dataset_id": "raw/missing.csv", "config": {}})
    assert r.status_code == 404


def test_training_failure_is_captured_in_job(trained_client: TestClient) -> None:
    r = trained_client.post("/training/start", json={"dataset_id": "synthetic/ci.csv",
                                                    "config": {"target_column": "no_such_col", "register": False}})
    job = trained_client.get(f"/training/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "failed" and "no_such_col" in job["error"]


# ---------------------------------------------------------------- registry
def test_versions_runs_and_aliases(trained_client: TestClient) -> None:
    versions = trained_client.get("/models/versions").json()
    assert [v["version"] for v in versions[:2]] == [2, 1]
    assert "champion" in next(v for v in versions if v["version"] == 1)["aliases"]
    assert trained_client.get("/models/runs").json()
    assert trained_client.post("/models/versions/2/alias", json={"alias": "challenger"}).json() == {"alias": "challenger", "version": 2}
    assert trained_client.post("/models/versions/99/alias", json={"alias": "x"}).status_code == 404
    assert trained_client.post("/models/versions/1/alias", json={"alias": "bad alias!"}).status_code == 422
    assert trained_client.delete("/models/aliases/challenger").json() == {"deleted": "challenger"}


def test_alias_listing_reflects_registry(trained_client: TestClient) -> None:
    """Regression: ``search_model_versions`` returns no aliases on the SQL store."""
    trained_client.post("/models/versions/2/alias", json={"alias": "shadow"})
    by_version = {v["version"]: v["aliases"] for v in trained_client.get("/models/versions").json()}
    assert "shadow" in by_version[2] and "champion" in by_version[1]
    trained_client.delete("/models/aliases/shadow")
    assert "shadow" not in {a for v in trained_client.get("/models/versions").json() for a in v["aliases"]}


def test_model_info_and_schema(trained_client: TestClient) -> None:
    info = trained_client.get("/model/info").json()
    assert info["name"] == "TelecomChurnClassifier" and info["feature_importance"] and info["summary"]["test_metrics"]
    schema = trained_client.get("/model/schema").json()
    kinds = {f["kind"] for f in schema["features"]}
    assert kinds == {"numeric", "categorical"} and 0 < schema["threshold"] < 1


# -------------------------------------------------------------- prediction
def test_predict_single_contract(trained_client: TestClient) -> None:
    res = trained_client.post("/predict", json={"features": feature_record(trained_client)})
    body = res.json()
    assert res.status_code == 200
    assert 0.0 <= body["churn_probability"] <= 1.0 and body["prediction"] in (0, 1)
    assert body["model_version"] == 1 and body["model_alias"] == "champion" and body["latency_ms"] >= 0


def test_predict_threshold_controls_label(trained_client: TestClient) -> None:
    rec = feature_record(trained_client)
    low = trained_client.post("/predict", json={"features": rec, "threshold": 0.001}).json()
    high = trained_client.post("/predict", json={"features": rec, "threshold": 0.999}).json()
    assert low["prediction"] == 1 and high["prediction"] == 0


def test_predict_imputes_a_few_missing_fields(trained_client: TestClient) -> None:
    rec = feature_record(trained_client)
    for k in list(rec)[:3]:
        rec.pop(k)
    body = trained_client.post("/predict", json={"features": rec}).json()
    assert any("Imputed 3" in w for w in body["warnings"])


def test_predict_rejects_too_many_missing_and_unknown_fields(trained_client: TestClient) -> None:
    rec = feature_record(trained_client)
    assert trained_client.post("/predict", json={"features": {list(rec)[0]: rec[list(rec)[0]]}}).status_code == 422
    assert trained_client.post("/predict", json={"features": {**rec, "hacker_field": 1}}).status_code == 422
    assert trained_client.post("/predict", json={"features": {}}).status_code == 422
    assert trained_client.post("/predict", json={"features": rec, "threshold": 1.5}).status_code == 422
    assert trained_client.post("/predict", json={"features": rec, "extra": 1}).status_code == 422


def test_predict_rejects_negative_tenure(trained_client: TestClient) -> None:
    rec = feature_record(trained_client)
    rec["tenure_months"] = -5
    r = trained_client.post("/predict", json={"features": rec})
    assert r.status_code == 422 and "tenure_months" in r.json()["detail"]


def test_batch_predict_and_upload(trained_client: TestClient) -> None:
    df = pd.read_csv(io.StringIO(trained_client.app.state.services.datasets.resolve("synthetic/ci.csv").read_text()))
    feats = trained_client.get("/model/schema").json()["features"]
    frame = df[[f["name"] for f in feats]].head(40)
    records = json.loads(frame.to_json(orient="records"))  # NaN -> null (strict JSON)
    body = trained_client.post("/predict/batch", json={"records": records}).json()
    assert body["n"] == 40 and len(body["predictions"]) == 40 and 0 <= body["positive_rate"] <= 1
    up = trained_client.post("/predict/batch/upload", params={"threshold": 0.4},
                             files={"file": ("b.csv", frame.to_csv(index=False).encode())}).json()
    assert up["n"] == 40 and up["threshold"] == 0.4


def test_batch_requires_every_feature(trained_client: TestClient) -> None:
    rec = feature_record(trained_client)
    rec.pop(next(iter(rec)))
    assert trained_client.post("/predict/batch", json={"records": [rec]}).status_code == 422
    assert trained_client.post("/predict/batch", json={"records": []}).status_code == 422


def test_sensitivity_numeric_and_categorical(trained_client: TestClient) -> None:
    feats = {f["name"]: f for f in trained_client.get("/model/schema").json()["features"]}
    num = next(n for n, f in feats.items() if f["kind"] == "numeric" and f["min"] != f["max"])
    cat = next(n for n, f in feats.items() if f["kind"] == "categorical" and len(f["categories"]) > 1)
    a = trained_client.post("/predict/sensitivity", json={"feature": num, "n_points": 7}).json()
    assert a["kind"] == "numeric" and len(a["values"]) == len(a["probabilities"]) == 7
    b = trained_client.post("/predict/sensitivity", json={"feature": cat}).json()
    assert b["kind"] == "categorical" and b["values"] == feats[cat]["categories"]
    assert all(0 <= p <= 1 for p in a["probabilities"] + b["probabilities"])
    assert trained_client.post("/predict/sensitivity", json={"feature": "nope"}).status_code == 422
    assert trained_client.post("/predict/sensitivity", json={"feature": num, "n_points": 1}).status_code == 422


# -------------------------------------------------------------- monitoring
def test_monitoring_endpoints_after_traffic(trained_client: TestClient) -> None:
    trained_client.post("/monitoring/reset")
    assert trained_client.get("/monitoring/summary").json()["window_predictions"] == 0
    assert trained_client.get("/monitoring/drift").json()["overall_status"] == "insufficient_data"
    sim = trained_client.post("/monitoring/simulate", json={"n": 300, "random_state": 5}).json()
    assert sim["simulated"] == 300
    summary = trained_client.get("/monitoring/summary").json()
    assert summary["window_predictions"] == 300 and summary["latency_ms"]["p95"] >= 0
    assert summary["expected_positive_rate"] is not None
    ts = trained_client.get("/monitoring/timeseries", params={"bucket_seconds": 5}).json()
    assert ts["buckets"] and "rolling_churn_rate" in ts["buckets"][0]
    drift = trained_client.get("/monitoring/drift").json()
    assert drift["overall_status"] in {"stable", "warning", "drift"} and drift["features"]
    assert drift["prediction_shift"]["psi"] is not None
    assert len(trained_client.get("/monitoring/recent", params={"n": 10}).json()) == 10


def test_drift_is_flagged_for_shifted_traffic(trained_client: TestClient) -> None:
    trained_client.post("/monitoring/reset")
    trained_client.post("/monitoring/simulate", json={"n": 400, "drift": True, "random_state": 9})
    drift = trained_client.get("/monitoring/drift").json()
    assert drift["overall_status"] in {"warning", "drift"}
    assert any(v["status"] == "drift" for v in drift["features"].values())


def test_single_predictions_are_logged(trained_client: TestClient) -> None:
    trained_client.post("/monitoring/reset")
    trained_client.post("/predict", json={"features": feature_record(trained_client)})
    assert trained_client.get("/monitoring/summary").json()["window_predictions"] == 1
    assert trained_client.get("/monitoring/recent").json()[0]["model_version"] == 1
