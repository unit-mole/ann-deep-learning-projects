from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SEED = 42
TARGET_COLUMN = "default_flag"
TRUE_PROBABILITY_COLUMN = "probability_default_true"
CLASSIFICATION_THRESHOLD = 0.58
LOW_RISK_UPPER = 0.20
MEDIUM_RISK_UPPER = 0.50

NUMERIC_FEATURES = [
    "age", "annual_income", "monthly_income", "loan_amount", "interest_rate",
    "employment_length", "credit_history_years", "revolving_utilization", "dti",
    "delinquency_count", "inquiry_count", "open_accounts", "existing_loans",
    "collateral_value", "loan_to_income", "collateral_ratio",
]

CATEGORICAL_FEATURES = [
    "home_ownership", "purpose", "grade", "region", "verification_status",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RAW_REQUIRED_COLUMNS = [
    "age", "annual_income", "loan_amount", "interest_rate", "employment_length",
    "credit_history_years", "revolving_utilization", "dti", "delinquency_count",
    "inquiry_count", "open_accounts", "existing_loans", "collateral_value",
    "home_ownership", "purpose", "grade", "region", "verification_status",
]
