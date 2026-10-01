"""Synthetic telecom subscriber generator and imbalance-handling helpers.

The generator produces ~20k subscribers with 40+ features (demographics,
services, financials, usage, support history). Churn is drawn from a
non-linear logit with interaction terms; the intercept is calibrated by
bisection so the realised churn rate matches ``target_churn_rate``.

Oversampling is exposed as *sampler factories* to be inserted into an
``imblearn.pipeline.Pipeline`` so that resampling only ever touches training
folds (never validation / test data).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

from ml_core.config import Settings, get_settings

SamplingStrategy = Literal["none", "class_weight", "random_over", "smote"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _calibrate_intercept(base_logit: np.ndarray, target_rate: float) -> float:
    """Find the intercept making ``mean(sigmoid(base + b)) == target_rate`` (bisection)."""
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(base_logit + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate_synthetic_subscribers(
    n_samples: int = 20_000,
    target_churn_rate: float = 0.20,
    random_state: int = 42,
    missing_rate: float = 0.0,
) -> pd.DataFrame:
    """Generate realistic synthetic telecom subscribers.

    Args:
        n_samples: Number of subscriber rows.
        target_churn_rate: Desired churn share (0.15-0.25 recommended).
        random_state: Seed (fully deterministic output).
        missing_rate: Fraction of cells in selected columns replaced by ``NaN``.

    Returns:
        DataFrame with an ID column, 40+ features and a binary ``churn`` column.

    Raises:
        ValueError: If ``n_samples`` or ``target_churn_rate`` are out of range.

    Example:
        >>> df = generate_synthetic_subscribers(1000, random_state=1)
        >>> df.shape[0], 0.1 < df["churn"].mean() < 0.35
        (1000, True)
    """
    if n_samples < 100:
        raise ValueError("n_samples must be >= 100.")
    if not 0.01 <= target_churn_rate <= 0.6:
        raise ValueError("target_churn_rate must be within [0.01, 0.6].")
    rng = np.random.default_rng(random_state)
    n = n_samples

    def pick(options: Sequence[str], p: Sequence[float]) -> np.ndarray:
        return rng.choice(np.asarray(options, dtype=object), size=n, p=p)

    # ---- demographics -----------------------------------------------------
    age = np.clip(rng.normal(44, 15, n), 18, 85).astype(int)
    gender = pick(["Male", "Female"], [0.5, 0.5])
    region = pick(["North", "South", "East", "West", "Central", "Coastal"], [0.18, 0.2, 0.15, 0.17, 0.15, 0.15])
    area_type = pick(["Urban", "Suburban", "Rural"], [0.5, 0.3, 0.2])
    marital_status = pick(["Single", "Married", "Divorced", "Widowed"], [0.35, 0.5, 0.11, 0.04])
    has_dependents = pick(["Yes", "No"], [0.3, 0.7])
    household_size = np.clip(rng.poisson(1.6, n) + 1, 1, 8)
    income_bracket = pick(["Low", "Lower-Mid", "Mid", "Upper-Mid", "High"], [0.15, 0.25, 0.3, 0.2, 0.1])

    # ---- services ---------------------------------------------------------
    contract_type = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.22, 0.23])
    tenure_months = np.minimum(rng.exponential(28, n).astype(int) + 1, 72)
    tenure_months = np.where(contract_type == "Two year", np.maximum(tenure_months, 12), tenure_months)
    plan_tier = pick(["Basic", "Standard", "Premium"], [0.35, 0.45, 0.2])
    phone_service = pick(["Yes", "No"], [0.9, 0.1])
    multiple_lines = np.where(phone_service == "No", "No phone service", pick(["Yes", "No"], [0.42, 0.58]))
    internet_service = pick(["None", "DSL", "Fiber", "5G Home"], [0.16, 0.28, 0.4, 0.16])
    no_net = internet_service == "None"

    def addon(p_yes: float) -> np.ndarray:
        return np.where(no_net, "No internet service", rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes]))

    online_security, online_backup = addon(0.3), addon(0.34)
    device_protection, tech_support = addon(0.34), addon(0.29)
    streaming_tv, streaming_movies = addon(0.38), addon(0.39)
    num_addons = sum((a == "Yes").astype(int) for a in (
        online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies))
    handset_brand = pick(["Apple", "Samsung", "Xiaomi", "Google", "Other"], [0.35, 0.3, 0.15, 0.08, 0.12])
    handset_age_months = np.clip(rng.gamma(3.0, 8.0, n), 1, 72).astype(int)

    # ---- financial --------------------------------------------------------
    base_price = np.select(
        [internet_service == "None", internet_service == "DSL", internet_service == "Fiber"], [22, 42, 68], default=58
    ).astype(float)
    tier_add = np.select([plan_tier == "Basic", plan_tier == "Standard"], [0, 8], default=18)
    monthly_charges = np.round(np.clip(base_price + tier_add + 4.5 * num_addons + rng.normal(0, 4, n), 15, 160), 2)
    total_charges = np.round(monthly_charges * tenure_months * rng.uniform(0.93, 1.05, n), 2)
    payment_method = pick(["Credit card", "Bank transfer", "Electronic check", "Mailed check"], [0.28, 0.27, 0.28, 0.17])
    autopay = np.where(np.isin(payment_method, ["Credit card", "Bank transfer"]), "Yes", "No")
    paperless_billing = pick(["Yes", "No"], [0.6, 0.4])
    late_payments_12m = np.clip(rng.poisson(np.where(income_bracket == "Low", 1.2, 0.5), n), 0, 12)
    avg_bill_change_pct = np.round(rng.normal(2.0, 9.0, n), 2)
    discount_active = pick(["Yes", "No"], [0.22, 0.78])
    promo_expiring_30d = np.where(discount_active == "Yes", rng.choice(["Yes", "No"], n, p=[0.35, 0.65]), "No")
    credit_score_band = pick(["Poor", "Fair", "Good", "Excellent"], [0.12, 0.28, 0.4, 0.2])

    # ---- usage ------------------------------------------------------------
    data_gb = np.where(no_net, rng.gamma(2, 2.5, n), rng.gamma(3.5, 28, n) * (1 + 0.15 * (plan_tier == "Premium")))
    avg_monthly_gb = np.round(data_gb, 2)
    avg_monthly_minutes = np.round(np.where(phone_service == "No", 0, rng.gamma(3.0, 140, n)), 1)
    avg_monthly_sms = rng.poisson(np.clip(90 - 0.6 * (age - 18), 5, None), n)
    roaming_minutes = np.round(rng.exponential(6, n) * (rng.random(n) < 0.25), 1)
    intl_calls_per_month = rng.poisson(0.8, n) * (phone_service == "Yes")
    data_overage_events_6m = rng.poisson(np.where(plan_tier == "Basic", 1.1, 0.3), n)
    usage_trend_3m_pct = np.round(rng.normal(-1.0, 22.0, n), 2)
    dropped_call_rate_pct = np.round(np.clip(rng.gamma(2.0, 0.9, n) * np.where(area_type == "Rural", 1.6, 1.0), 0, 25), 2)
    avg_download_mbps = np.round(np.clip(rng.normal(45, 20, n) + 25 * (internet_service == "Fiber"), 1, 400), 1)
    days_since_last_topup = np.clip(rng.exponential(18, n), 0, 180).astype(int)

    # ---- support history --------------------------------------------------
    support_calls_90d = rng.poisson(0.9 + 0.35 * (dropped_call_rate_pct > 4) + 0.2 * (internet_service == "Fiber"), n)
    complaints_12m = rng.poisson(0.25 + 0.12 * support_calls_90d, n)
    avg_resolution_hours = np.round(np.clip(rng.gamma(2.2, 14, n), 0.5, 240), 1)
    technician_visits_6m = rng.poisson(0.35, n)
    outage_incidents_6m = rng.poisson(0.6 + 0.4 * (area_type == "Rural"), n)
    nps_score = np.clip(np.round(8 - 0.5 * complaints_12m - 0.04 * dropped_call_rate_pct * 3 + rng.normal(0, 1.8, n)), 0, 10).astype(int)
    last_contact_days_ago = np.clip(rng.exponential(55, n), 0, 365).astype(int)
    retention_offers_received = rng.poisson(0.35, n)

    # ---- churn logit with non-linear interactions ---------------------------
    is_m2m = contract_type == "Month-to-month"
    logit = (
        1.1 * is_m2m - 0.35 * (contract_type == "One year") - 1.1 * (contract_type == "Two year")
        - 0.035 * (tenure_months - 24) + 0.9 * (tenure_months < 6)
        + 0.6 * (internet_service == "Fiber") * (monthly_charges - 70) / 30
        + 0.25 * support_calls_90d + 0.35 * complaints_12m * (avg_resolution_hours > 48)
        - 0.15 * (nps_score - 7)
        + 0.8 * (promo_expiring_30d == "Yes") * is_m2m
        - 0.015 * usage_trend_3m_pct + 0.7 * (usage_trend_3m_pct < -30)
        + 0.10 * dropped_call_rate_pct + 0.18 * late_payments_12m
        - 0.35 * (autopay == "Yes") - 0.35 * (tech_support == "Yes") - 0.25 * (online_security == "Yes")
        + 0.0007 * (age - 45) ** 2 - 0.3 * (retention_offers_received > 0)
        + 0.12 * outage_incidents_6m + 0.02 * np.clip(avg_bill_change_pct, 0, None)
        + 0.25 * (handset_age_months > 36) * (handset_brand != "Apple")
        + 0.3 * (data_overage_events_6m >= 3) + rng.normal(0, 0.5, n)
    ).astype(float)
    logit += _calibrate_intercept(logit, target_churn_rate)
    churn = (rng.random(n) < _sigmoid(logit)).astype(int)

    df = pd.DataFrame({
        "subscriber_id": [f"SUB-{i:07d}" for i in range(n)],
        "age": age, "gender": gender, "region": region, "area_type": area_type,
        "marital_status": marital_status, "has_dependents": has_dependents,
        "household_size": household_size, "income_bracket": income_bracket,
        "contract_type": contract_type, "tenure_months": tenure_months, "plan_tier": plan_tier,
        "phone_service": phone_service, "multiple_lines": multiple_lines,
        "internet_service": internet_service, "online_security": online_security,
        "online_backup": online_backup, "device_protection": device_protection,
        "tech_support": tech_support, "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies, "num_addon_services": num_addons,
        "handset_brand": handset_brand, "handset_age_months": handset_age_months,
        "monthly_charges": monthly_charges, "total_charges": total_charges,
        "payment_method": payment_method, "autopay": autopay,
        "paperless_billing": paperless_billing, "late_payments_12m": late_payments_12m,
        "avg_bill_change_pct": avg_bill_change_pct, "discount_active": discount_active,
        "promo_expiring_30d": promo_expiring_30d, "credit_score_band": credit_score_band,
        "avg_monthly_gb": avg_monthly_gb, "avg_monthly_minutes": avg_monthly_minutes,
        "avg_monthly_sms": avg_monthly_sms, "roaming_minutes": roaming_minutes,
        "data_overage_events_6m": data_overage_events_6m,
        "usage_trend_3m_pct": usage_trend_3m_pct, "dropped_call_rate_pct": dropped_call_rate_pct,
        "avg_download_mbps": avg_download_mbps, "days_since_last_topup": days_since_last_topup,
        "support_calls_90d": support_calls_90d, "complaints_12m": complaints_12m,
        "avg_resolution_hours": avg_resolution_hours, "technician_visits_6m": technician_visits_6m,
        "outage_incidents_6m": outage_incidents_6m, "nps_score": nps_score,
        "last_contact_days_ago": last_contact_days_ago,
        "retention_offers_received": retention_offers_received, "churn": churn,
    })
    if missing_rate > 0:
        df = inject_missing_values(df, missing_rate, random_state=random_state)
    return df


def inject_missing_values(
    df: pd.DataFrame, rate: float, columns: Sequence[str] | None = None, random_state: int = 42
) -> pd.DataFrame:
    """Randomly blank out cells (MCAR) in feature columns.

    Args:
        df: Source dataframe (not mutated).
        rate: Fraction of cells to blank per selected column (0-1).
        columns: Columns to affect; defaults to a fixed set of numeric/categorical features.
        random_state: Seed.

    Returns:
        Copy with missing values injected. ID and target columns are never touched.

    Raises:
        ValueError: If ``rate`` is outside [0, 1].
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be within [0, 1].")
    rng = np.random.default_rng(random_state + 1)
    out = df.copy()
    default_cols = ["monthly_charges", "avg_monthly_gb", "nps_score", "income_bracket",
                    "credit_score_band", "avg_download_mbps", "handset_brand", "usage_trend_3m_pct"]
    for col in columns or [c for c in default_cols if c in out.columns]:
        if col in {"churn", "subscriber_id"} or col not in out.columns:
            continue
        mask = rng.random(len(out)) < rate
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(float).where(~mask, np.nan)
        else:
            out[col] = out[col].astype("object").where(~mask, np.nan)
    return out


def save_synthetic_dataset(
    df: pd.DataFrame, filename: str = "synthetic_subscribers.parquet", settings: Settings | None = None
) -> Path:
    """Persist a synthetic dataset under ``data/synthetic``.

    Args:
        df: Dataset to store.
        filename: Target file name (``.csv`` or ``.parquet``).
        settings: Optional settings override.

    Returns:
        Absolute path of the written file.

    Raises:
        ValueError: If the extension is not ``.csv`` / ``.parquet``.
    """
    cfg = settings or get_settings()
    cfg.data_synthetic_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.data_synthetic_dir / filename
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("filename must end with .csv or .parquet")
    return path


# --------------------------------------------------------------------------- #
# Imbalance handling
# --------------------------------------------------------------------------- #
def uses_class_weight(strategy: SamplingStrategy) -> bool:
    """Whether the strategy is algorithm-level cost-sensitive learning."""
    return strategy == "class_weight"


def build_sampler(
    strategy: SamplingStrategy, *, sampling_ratio: float = 0.6, random_state: int = 42,
    k_neighbors: int = 5, minority_ratio: float | None = None,
) -> BaseSampler | None:
    """Create an imblearn sampler to embed in ``imblearn.pipeline.Pipeline``.

    Args:
        strategy: ``"smote"`` or ``"random_over"`` return a sampler; ``"none"`` and
            ``"class_weight"`` return ``None`` (the latter is applied in the classifier).
        sampling_ratio: Desired minority/majority ratio after resampling.
        random_state: Seed.
        k_neighbors: SMOTE neighbourhood size.
        minority_ratio: Current minority/majority ratio. If it already reaches
            ``sampling_ratio`` no resampling is needed and ``None`` is returned.

    Returns:
        Sampler instance or ``None``.

    Raises:
        ValueError: If ``strategy`` is unknown.
    """
    if strategy in ("none", "class_weight"):
        return None
    if strategy not in ("smote", "random_over"):
        raise ValueError(f"Unknown sampling strategy '{strategy}'.")
    if minority_ratio is not None and minority_ratio >= sampling_ratio - 0.02:
        return None
    if strategy == "smote":
        return SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k_neighbors, random_state=random_state)
    return RandomOverSampler(sampling_strategy=sampling_ratio, random_state=random_state)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    frame = generate_synthetic_subscribers()
    print(f"Wrote {save_synthetic_dataset(frame)} | churn rate={frame['churn'].mean():.3f} | cols={frame.shape[1]}")
