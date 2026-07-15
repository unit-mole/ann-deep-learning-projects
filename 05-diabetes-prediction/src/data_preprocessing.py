"""Data loading, splitting, and leakage-safe preprocessing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN
from .feature_engineering import invalid_zero_counts, prepare_feature_frame


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the project dataset."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    unique_targets = set(pd.to_numeric(data[TARGET_COLUMN], errors="coerce").dropna())
    if not unique_targets.issubset({0, 1}):
        raise ValueError("Outcome must contain binary values 0 and 1 only.")

    return data


def split_dataset(
    data: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train, validation, and test partitions."""
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to < 1.")

    X = prepare_feature_frame(data)
    y = data[TARGET_COLUMN].astype(int)

    holdout_size = test_size + validation_size
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        stratify=y,
        random_state=random_state,
    )
    relative_test_size = test_size / holdout_size
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=relative_test_size,
        stratify=y_holdout,
        random_state=random_state,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_preprocessor() -> Pipeline:
    """Build median imputation plus standardization.

    Missing-value indicators are included so the ANN can learn whether a source
    value was absent without treating a physiologically implausible zero as a
    genuine measurement.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )


def create_data_quality_report(data: pd.DataFrame) -> dict[str, Any]:
    """Summarize class balance, missingness, and zero-as-missing markers."""
    class_counts = data[TARGET_COLUMN].value_counts().sort_index().to_dict()
    return {
        "rows": int(len(data)),
        "feature_count": len(FEATURE_COLUMNS),
        "class_counts": {str(int(k)): int(v) for k, v in class_counts.items()},
        "positive_rate": float(data[TARGET_COLUMN].mean()),
        "explicit_missing_values": {
            column: int(count) for column, count in data.isna().sum().items()
        },
        "zero_as_missing_counts": invalid_zero_counts(data),
    }
