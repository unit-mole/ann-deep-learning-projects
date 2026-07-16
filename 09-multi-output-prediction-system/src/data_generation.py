from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import SEED


def generate_synthetic_customer_data(n_samples: int = 15_000, seed: int = SEED) -> pd.DataFrame:
    """Generate the reproducible synthetic customer dataset used by the project."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.randint(18, 75, n_samples),
        "tenure_months": rng.randint(1, 121, n_samples),
        "monthly_income": rng.normal(6500, 2200, n_samples).clip(1200, 25000),
        "monthly_spend": rng.normal(420, 180, n_samples).clip(20, 2500),
        "transactions_last_30d": rng.poisson(8, n_samples).clip(0, 50),
        "avg_session_minutes": rng.normal(24, 10, n_samples).clip(1, 120),
        "products_owned": rng.randint(1, 8, n_samples),
        "complaints_last_6m": rng.poisson(0.8, n_samples).clip(0, 10),
        "days_since_last_login": rng.randint(0, 120, n_samples),
        "discount_usage_rate": rng.beta(2.2, 4.5, n_samples),
        "support_tickets_last_90d": rng.poisson(1.5, n_samples).clip(0, 20),
        "website_visits_last_30d": rng.poisson(14, n_samples).clip(0, 80),
        "region": rng.choice(["North", "South", "East", "West", "Central"], n_samples, p=[.22, .18, .20, .23, .17]),
        "channel": rng.choice(["Web", "Mobile", "Store", "Partner"], n_samples, p=[.38, .34, .17, .11]),
        "plan_type": rng.choice(["Basic", "Standard", "Premium", "Enterprise"], n_samples, p=[.36, .33, .22, .09]),
        "segment": rng.choice(["Budget", "Growth", "Loyal", "HighValue"], n_samples, p=[.30, .28, .24, .18]),
        "device_type": rng.choice(["Android", "iOS", "Desktop", "Tablet"], n_samples, p=[.34, .28, .28, .10]),
    })
    plan = df["plan_type"].map({"Basic": .85, "Standard": 1.0, "Premium": 1.2, "Enterprise": 1.45}).astype(float)
    segment = df["segment"].map({"Budget": .85, "Growth": 1.0, "Loyal": 1.15, "HighValue": 1.35}).astype(float)
    channel = df["channel"].map({"Web": 1.0, "Mobile": 1.05, "Store": .95, "Partner": 1.08}).astype(float)
    engagement_signal = (
        .30 * df["website_visits_last_30d"] + .45 * df["avg_session_minutes"]
        + .25 * df["transactions_last_30d"] - .35 * df["days_since_last_login"]
        - 1.2 * df["complaints_last_6m"] - .7 * df["support_tickets_last_90d"]
        + 2.0 * df["products_owned"]
    )
    df["engagement_score"] = (45 + engagement_signal + 5 * plan + 3 * segment + rng.normal(0, 6, n_samples)).clip(1, 100)
    df["clv"] = (
        12 * df["monthly_spend"] * plan * segment + .9 * df["monthly_income"]
        + 35 * df["tenure_months"] + 40 * df["transactions_last_30d"]
        + 15 * df["website_visits_last_30d"] - 140 * df["complaints_last_6m"]
        - 80 * df["support_tickets_last_90d"] + 250 * channel + rng.normal(0, 900, n_samples)
    ).clip(100, 50000)
    logit = (
        -2.20 - .018 * df["tenure_months"] - .030 * df["engagement_score"]
        - .00020 * df["clv"] + .030 * df["days_since_last_login"]
        + .280 * df["complaints_last_6m"] + .180 * df["support_tickets_last_90d"]
        - .060 * df["transactions_last_30d"] + .90 * (df["plan_type"] == "Basic").astype(int)
        + .35 * (df["segment"] == "Budget").astype(int) + rng.normal(0, .6, n_samples)
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    df["churn"] = (rng.rand(n_samples) < churn_probability).astype(int)
    return inject_missingness(df, seed=seed)


def inject_missingness(df: pd.DataFrame, rate: float = 0.02, seed: int = SEED) -> pd.DataFrame:
    result = df.copy()
    rng = np.random.default_rng(seed)
    for column in ["monthly_income", "monthly_spend", "avg_session_minutes", "region", "channel"]:
        indices = rng.choice(result.index, size=int(rate * len(result)), replace=False)
        result.loc[indices, column] = np.nan
    return result
