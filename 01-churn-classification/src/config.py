from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
ASSET_DIR = PROJECT_ROOT / "assets"

IDENTIFIER_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET_COLUMN = "Exited"

REQUIRED_FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

CATEGORICAL_FEATURES = ["Geography", "Gender"]

ALLOWED_GEOGRAPHIES = {"France", "Germany", "Spain"}
ALLOWED_GENDERS = {"Female", "Male"}

NUMERIC_RANGES = {
    "CreditScore": (300, 900),
    "Age": (18, 100),
    "Tenure": (0, 10),
    "Balance": (0.0, 300000.0),
    "NumOfProducts": (1, 4),
    "HasCrCard": (0, 1),
    "IsActiveMember": (0, 1),
    "EstimatedSalary": (0.0, 250000.0),
}
