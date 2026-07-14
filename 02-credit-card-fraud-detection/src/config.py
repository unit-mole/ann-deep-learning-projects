"""Shared project paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_COLUMN = "Class"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

DEFAULT_MODEL_PATH = MODEL_DIR / "credit_card_fraud_ann.keras"
DEFAULT_SCALER_PATH = MODEL_DIR / "credit_card_scaler.pkl"
DEFAULT_THRESHOLD_PATH = MODEL_DIR / "decision_threshold.json"
DEFAULT_SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
