"""Data loading, validation, splitting, scaling, and outlier diagnostics."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN
except ImportError:  # Allows direct execution from the src directory.
    from config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN


def load_housing_data(path: str | Path) -> pd.DataFrame:
    """Load the project CSV and validate its required schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    validate_dataset(data)
    return data


def validate_dataset(data: pd.DataFrame) -> None:
    """Raise a clear error when required columns or usable values are missing."""
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    non_numeric = [
        column for column in required if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric:
        raise TypeError(f"Expected numeric columns, found non-numeric data in: {non_numeric}")

    if data[required].isna().all(axis=0).any():
        empty_columns = data[required].columns[data[required].isna().all()].tolist()
        raise ValueError(f"Columns contain no usable values: {empty_columns}")


def impute_numeric_values(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Median-impute numeric features using training-set statistics only."""
    medians = train[FEATURE_COLUMNS].median().to_dict()

    def _fill(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output[FEATURE_COLUMNS] = output[FEATURE_COLUMNS].fillna(medians)
        return output

    return _fill(train), _fill(validation), _fill(test), {
        key: float(value) for key, value in medians.items()
    }


def split_housing_data(
    data: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create deterministic 70%/15%/15% train, validation, and test splits."""
    features = data[FEATURE_COLUMNS].copy()
    target = data[TARGET_COLUMN].copy()

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=random_state,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_iqr_outlier_report(data: pd.DataFrame) -> pd.DataFrame:
    """Report IQR outliers without deleting observations."""
    rows: list[dict[str, float | int | str]] = []
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        series = data[column].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        rows.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower,
                "upper_fence": upper,
                "outlier_count": count,
                "outlier_percent": 100.0 * count / len(series),
            }
        )
    return pd.DataFrame(rows)


def training_quantile_bounds(
    x_train: pd.DataFrame,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> dict[str, dict[str, float]]:
    """Calculate optional train-only clipping bounds."""
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")

    bounds: dict[str, dict[str, float]] = {}
    for column in FEATURE_COLUMNS:
        bounds[column] = {
            "lower": float(x_train[column].quantile(lower_quantile)),
            "upper": float(x_train[column].quantile(upper_quantile)),
        }
    return bounds


def clip_to_bounds(
    frame: pd.DataFrame,
    bounds: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Clip features to precomputed train-only bounds."""
    output = frame.copy()
    for column in FEATURE_COLUMNS:
        if column in bounds:
            output[column] = output[column].clip(
                lower=float(bounds[column]["lower"]),
                upper=float(bounds[column]["upper"]),
            )
    return output


def fit_scaler(x_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler using training features only."""
    scaler = StandardScaler()
    scaler.fit(x_train[FEATURE_COLUMNS])
    return scaler


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    """Persist the fitted scaler."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(scaler, file)
