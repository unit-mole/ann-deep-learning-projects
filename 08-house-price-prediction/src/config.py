"""Central project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DATA_PATH = DATA_DIR / "house_prices.csv"
SAMPLE_INPUT_PATH = DATA_DIR / "sample_input.csv"

MODEL_PATH = MODEL_DIR / "house_price_ann.keras"
SCALER_PATH = MODEL_DIR / "house_price_scaler.pkl"
PARAMS_PATH = MODEL_DIR / "house_price_best_params.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

TARGET_COLUMN = "SalePrice"
FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_MULTIPLIER = 100_000
RANDOM_STATE = 42
