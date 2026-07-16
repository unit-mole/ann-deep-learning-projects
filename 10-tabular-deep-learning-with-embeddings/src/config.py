from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"

TARGET_COLUMN = "target"
CATEGORICAL_FEATURES = [
    "region",
    "education_level",
    "occupation",
    "channel",
    "device_type",
    "membership_tier",
    "product_category",
]
NUMERICAL_FEATURES = [
    "age",
    "years_experience",
    "monthly_sessions",
    "avg_basket_value",
    "days_since_last_activity",
    "credit_utilization",
    "income_estimate",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
SEED = 42
