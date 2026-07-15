"""Central project configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"

RANDOM_STATE = 42
TARGET_COLUMN = "Outcome"
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# In the Pima dataset, zero is physiologically implausible for these fields
# and is treated as a missing-value marker. Zero pregnancies remains valid.
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

DEFAULT_RISK_THRESHOLDS = {
    "low_upper": 0.30,
    "high_lower": 0.60,
}
