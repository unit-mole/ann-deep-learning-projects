from __future__ import annotations

SEED = 42
RAW_FEATURES = [
    "age", "tenure_months", "monthly_income", "monthly_spend",
    "transactions_last_30d", "avg_session_minutes", "products_owned",
    "complaints_last_6m", "days_since_last_login", "discount_usage_rate",
    "support_tickets_last_90d", "website_visits_last_30d", "region",
    "channel", "plan_type", "segment", "device_type",
]
ENGINEERED_FEATURES = ["tenure_cohort", "spend_band"]
MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
NUMERIC_FEATURES = RAW_FEATURES[:12]
CATEGORICAL_FEATURES = RAW_FEATURES[12:] + ENGINEERED_FEATURES
TARGETS = ["churn", "clv", "engagement_score"]
