# Telecom Customer Churn — MLOps microservice

Production-oriented, fully decoupled churn-prediction system:

| Layer | Package | Responsibility |
|---|---|---|
| ML core | `ml_core/` | config, dynamic schema + quality gates, synthetic data, leak-free preprocessing, model factory, baseline/Optuna trainer, MLflow registry |
| REST API | `api/` | FastAPI app, hot-reloading model service, telemetry + drift monitor |
| Frontend | `streamlit_app/` | Streamlit dashboard — **REST only** (imports no sklearn / mlflow / `ml_core`) |
| Client | `notebooks/` | End-to-end API lifecycle notebook |
| Tests | `tests/` | pytest suite (data gates, pipeline, API contracts, hot reload) |

Everything lives in `app_ml_rest_api/`; the pre-existing `app/` directory (if any) is untouched.

## Quick start

```bash
cd app_ml_rest_api
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # adjust if needed

uvicorn api.main:app --port 8000          # API  -> http://localhost:8000/docs
streamlit run streamlit_app/app.py        # UI   -> http://localhost:8501
```

First run: open the dashboard → **EDA & PCA** → *Generate synthetic subscribers* (or upload the
[Kaggle Telco file](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) →
**Training & Registry** → train → *Promote to @champion* → *Reload model* → **Prediction**.

### Docker

```bash
cp .env.example .env
docker compose up --build        # api :8000, dashboard :8501
```

The multi-stage `Dockerfile` builds wheels in a builder stage, runs as a non-root user and
exposes a `/health` healthcheck. `data/` and `models/` are mounted as volumes so the MLflow DB,
artifacts and telemetry survive restarts.

## Configuration (`.env`)

All paths are resolved against the project root, never the working directory.

| Variable | Default | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///models/mlflow.db` | Relational registry DB (SQLite or `postgresql+psycopg2://…`) |
| `MLFLOW_ARTIFACT_ROOT` | `./models/artifacts/` | Artifact store (pipelines, plots, reference profiles) |
| `CHAMPION_ALIAS` / `CHALLENGER_ALIAS` | `champion` / `challenger` | Alias names |
| `MAX_MISSING_FRACTION` | `0.30` | Any feature above this missing share is rejected |
| `NON_NEGATIVE_PATTERNS` | tenure, charge, … | Regexes of numeric columns that must be ≥ 0 |
| `NUMERIC_IMPUTATION` | `median` | `median \| mean \| constant \| iterative` |
| `CATEGORICAL_IMPUTATION` | `most_frequent` | `most_frequent \| constant` |
| `SAMPLING_STRATEGY` | `class_weight` | `none \| class_weight \| random_over \| smote` |
| `PSI_WARNING` / `PSI_ALERT` | `0.10` / `0.25` | Drift thresholds |

See `.env.example` for the complete list. PostgreSQL: set the URI and uncomment `psycopg2-binary`.

## Design guarantees

* **No leakage** – split first; imputers, scalers, encoders and oversamplers sit in one
  `imblearn.pipeline.Pipeline` refitted per CV fold. Optuna only sees the training split; the
  test set is scored exactly once.
* **Dynamic schema** – column roles (numeric / categorical / id / constant / target) are inferred
  per dataset (incl. blank-string numerics such as Telco `TotalCharges`) and stored with each model.
* **Quality gates** – min rows, missing-% threshold, dtype conformity, unexpected / missing columns,
  non-negative ranges, minority-class size → `DataValidationError` (HTTP 422 with full report).
* **Zero-downtime hot reload** – candidate is downloaded, deserialised and dry-run *outside* the
  read lock, then swapped atomically. In-flight requests finish on the old immutable snapshot; a
  failing dry-run leaves the active model untouched.
* **No global mutable state** – services live on `app.state.services`.
* **Reproducible** – `random_state=42` is propagated everywhere.

## REST API (selected)

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness, DB check, serving model |
| `POST /datasets/upload` · `POST /datasets/synthetic` · `GET /datasets` | Data ingestion |
| `POST /datasets/validate` · `GET /datasets/{src}/{name}/eda\|pca\|columns/{col}` | Schema, gates, EDA, PCA |
| `POST /training/start` · `GET /training/jobs[/{id}]` | Background baseline / Optuna jobs |
| `GET /models/versions` · `POST /models/versions/{v}/alias\|promote` · `DELETE /models/aliases/{a}` | Registry & aliases |
| `POST /model/reload?alias=&version=` | Hot reload (default `@champion`) |
| `POST /predict` · `/predict/batch` · `/predict/batch/upload` · `/predict/sensitivity` | Inference |
| `GET /monitoring/summary\|timeseries\|drift\|recent` · `POST /monitoring/reset\|simulate` | Telemetry |

Interactive docs: `http://localhost:8000/docs`.

### Training payload example

```json
{
  "dataset_id": "synthetic/synthetic_20000_42.parquet",
  "config": {
    "model_name": "xgboost", "mode": "optuna", "n_trials": 30, "optimize_metric": "pr_auc",
    "sampling": "smote", "cv_folds": 5,
    "param_space": {"max_depth": {"type": "int", "low": 3, "high": 8}}
  }
}
```

## Monitoring

Every scored request is kept in a bounded in-memory window and appended to
`models/monitoring/inference_log.jsonl` (rotated at 50 MB, restored on start-up). The dashboard shows
request volume, rolling churn rate vs the training baseline, latency percentiles, per-feature
PSI + KS p-values and prediction-distribution PSI. Use *Send simulated traffic* (optionally with drift)
to exercise the dashboards.

## Tests

```bash
pytest -q                 # whole suite (trains two tiny models once per session, ~1 min)
pytest tests/test_data_quality.py -q       # fast, no training
```

All tests use temporary SQLite databases and artifact folders — the real `data/` and `models/`
directories are never touched.
