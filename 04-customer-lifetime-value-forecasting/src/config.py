from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = MODELS_DIR / "clv_ann_model.keras"
SCALER_PATH = MODELS_DIR / "numeric_scaler.pkl"
ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SEGMENT_SCALER_PATH = MODELS_DIR / "segment_scaler.pkl"
SEGMENTER_PATH = MODELS_DIR / "customer_segmenter.pkl"

NUMERIC_FEATURES = [
    "n_orders", "total_revenue", "avg_order_value", "std_order_value",
    "total_quantity", "avg_quantity", "category_diversity", "payment_diversity",
    "channel_diversity", "avg_discount", "recency_days", "tenure_days",
    "orders_per_month", "revenue_per_month", "loyalty_score",
    "discount_sensitivity", "engagement_score", "cohort_age_months",
    "avg_revenue_per_active_month",
]

CATEGORICAL_FEATURES = [
    "country", "preferred_channel", "dominant_category",
    "acquisition_quarter", "customer_segment_name",
]

SEGMENTATION_FEATURES = [
    "n_orders", "total_revenue", "avg_order_value", "recency_days", "category_diversity"
]

NON_NEGATIVE_FEATURES = set(NUMERIC_FEATURES)
PROPORTION_FEATURES = {"avg_discount", "discount_sensitivity"}
SCORE_FEATURES = {"loyalty_score", "engagement_score"}
INTEGER_LIKE_FEATURES = {
    "n_orders", "total_quantity", "category_diversity", "payment_diversity",
    "channel_diversity", "recency_days", "tenure_days",
}
