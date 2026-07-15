from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic model features when the raw inputs are available."""
    df = frame.copy()

    if "tenure_days" in df:
        tenure_months = pd.to_numeric(df["tenure_days"], errors="coerce").clip(lower=1) / 30.0
        if "cohort_age_months" not in df:
            df["cohort_age_months"] = tenure_months
    else:
        tenure_months = pd.Series(np.nan, index=df.index)

    if "orders_per_month" not in df and {"n_orders", "tenure_days"}.issubset(df.columns):
        df["orders_per_month"] = pd.to_numeric(df["n_orders"], errors="coerce") / tenure_months.clip(lower=1)

    if "revenue_per_month" not in df and {"total_revenue", "tenure_days"}.issubset(df.columns):
        df["revenue_per_month"] = pd.to_numeric(df["total_revenue"], errors="coerce") / tenure_months.clip(lower=1)

    if "avg_order_value" not in df and {"total_revenue", "n_orders"}.issubset(df.columns):
        orders = pd.to_numeric(df["n_orders"], errors="coerce").replace(0, np.nan)
        df["avg_order_value"] = pd.to_numeric(df["total_revenue"], errors="coerce") / orders

    if "avg_quantity" not in df and {"total_quantity", "n_orders"}.issubset(df.columns):
        orders = pd.to_numeric(df["n_orders"], errors="coerce").replace(0, np.nan)
        df["avg_quantity"] = pd.to_numeric(df["total_quantity"], errors="coerce") / orders

    if "avg_revenue_per_active_month" not in df and {"total_revenue", "cohort_age_months"}.issubset(df.columns):
        active_months = pd.to_numeric(df["cohort_age_months"], errors="coerce").clip(lower=1)
        df["avg_revenue_per_active_month"] = pd.to_numeric(df["total_revenue"], errors="coerce") / active_months

    return df


def infer_legacy_customer_segment(frame: pd.DataFrame) -> pd.Series:
    """Deployment fallback for the supplied model when its original K-means artifact is unavailable.

    The original notebook saved the segment label as a model input but did not save the
    fitted K-means scaler or estimator. These transparent rules approximate the original
    cluster profiles. Retraining with ``src/model_training.py`` saves the real artifacts.
    """
    df = add_derived_features(frame)
    n_orders = pd.to_numeric(df.get("n_orders", 0), errors="coerce").fillna(0)
    revenue = pd.to_numeric(df.get("total_revenue", 0), errors="coerce").fillna(0)
    recency = pd.to_numeric(df.get("recency_days", 999), errors="coerce").fillna(999)

    return pd.Series(
        np.select(
            [
                (recency >= 180) | (n_orders <= 3),
                (revenue >= 1300) & (n_orders >= 12) & (recency <= 90),
                (revenue >= 650) & (recency <= 120),
            ],
            ["Segment_1", "Segment_2", "Segment_0"],
            default="Segment_3",
        ),
        index=df.index,
        dtype="object",
    )


def build_customer_feature_table(
    transactions: pd.DataFrame,
    observation_end: str | pd.Timestamp = "2024-09-30",
    forecast_end: str | pd.Timestamp = "2024-12-29",
) -> pd.DataFrame:
    """Aggregate transaction rows to one customer snapshot with 90-day targets."""
    required = {
        "customer_id", "order_date", "order_value", "quantity", "product_category",
        "payment_type", "transaction_channel", "discount_rate", "country",
        "preferred_channel", "loyalty_score", "discount_sensitivity", "engagement_score",
    }
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(f"Transactions are missing required columns: {missing}")

    tx = transactions.copy()
    tx["order_date"] = pd.to_datetime(tx["order_date"], errors="raise")
    observation_end = pd.Timestamp(observation_end)
    forecast_end = pd.Timestamp(forecast_end)
    obs = tx[tx["order_date"] <= observation_end].copy()
    future = tx[(tx["order_date"] > observation_end) & (tx["order_date"] <= forecast_end)].copy()

    features = obs.groupby("customer_id").agg(
        n_orders=("order_value", "count"),
        total_revenue=("order_value", "sum"),
        avg_order_value=("order_value", "mean"),
        std_order_value=("order_value", "std"),
        total_quantity=("quantity", "sum"),
        avg_quantity=("quantity", "mean"),
        last_order_date=("order_date", "max"),
        first_order_date=("order_date", "min"),
        category_diversity=("product_category", "nunique"),
        payment_diversity=("payment_type", "nunique"),
        channel_diversity=("transaction_channel", "nunique"),
        avg_discount=("discount_rate", "mean"),
    ).reset_index()
    features["std_order_value"] = features["std_order_value"].fillna(0.0)
    features["avg_discount"] = features["avg_discount"].fillna(0.0)
    features["recency_days"] = (observation_end - features["last_order_date"]).dt.days
    features["tenure_days"] = (observation_end - features["first_order_date"]).dt.days + 1
    features = add_derived_features(features)

    static = (
        obs.sort_values("order_date")
        .groupby("customer_id", as_index=False)
        .agg(
            country=("country", "last"),
            preferred_channel=("preferred_channel", "last"),
            loyalty_score=("loyalty_score", "last"),
            discount_sensitivity=("discount_sensitivity", "last"),
            engagement_score=("engagement_score", "last"),
        )
    )
    features = features.merge(static, on="customer_id", how="left")

    dominant = (
        obs.groupby(["customer_id", "product_category"], as_index=False)["order_value"].sum()
        .sort_values(["customer_id", "order_value"], ascending=[True, False])
        .drop_duplicates("customer_id")
        .rename(columns={"product_category": "dominant_category"})
        [["customer_id", "dominant_category"]]
    )
    features = features.merge(dominant, on="customer_id", how="left")
    features["acquisition_quarter"] = features["first_order_date"].dt.to_period("Q").astype(str)
    cohort_start = features["first_order_date"].dt.to_period("M").dt.to_timestamp()
    features["cohort_age_months"] = ((observation_end - cohort_start).dt.days / 30.0).clip(lower=0.1)
    features["avg_revenue_per_active_month"] = features["total_revenue"] / features["cohort_age_months"].clip(lower=1)

    future_summary = future.groupby("customer_id").agg(
        future_90d_revenue=("order_value", "sum"),
        future_90d_orders=("order_value", "count"),
    ).reset_index()
    features = features.merge(future_summary, on="customer_id", how="left")
    features["future_90d_revenue"] = features["future_90d_revenue"].fillna(0.0)
    features["future_90d_orders"] = features["future_90d_orders"].fillna(0).astype(int)
    features["retained_90d"] = (features["future_90d_orders"] > 0).astype(int)
    return features
