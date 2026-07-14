"""Data validation, splitting, scaling, and sample-generation utilities."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN


@dataclass(frozen=True)
class DatasetSplits:
    """Container for stratified train, validation, and test splits."""

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the expected credit-card dataset from disk.

    The function deliberately does not silently download or substitute synthetic
    data. A portfolio project should fail clearly when the required source data
    is missing rather than train on a different dataset without the user's
    knowledge.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. See data/README_data.md for setup."
        )

    dataframe = pd.read_csv(path)
    validate_dataset(dataframe, require_target=True)
    return dataframe


def validate_dataset(
    dataframe: pd.DataFrame,
    *,
    require_target: bool = True,
    required_features: Iterable[str] = FEATURE_COLUMNS,
) -> None:
    """Validate columns, dtypes, missing values, and binary target values."""
    if dataframe.empty:
        raise ValueError("The supplied dataset is empty.")

    required = list(required_features)
    expected = required + ([TARGET_COLUMN] if require_target else [])
    missing_columns = [column for column in expected if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing_columns)
        )

    duplicated_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(
            "Duplicate column name(s) detected: " + ", ".join(duplicated_columns)
        )

    feature_frame = dataframe.loc[:, required]
    non_numeric = [
        column
        for column in required
        if not pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    if non_numeric:
        raise TypeError(
            "All model features must be numeric. Non-numeric column(s): "
            + ", ".join(non_numeric)
        )

    if feature_frame.isna().any().any():
        missing_counts = feature_frame.isna().sum()
        details = {
            column: int(count)
            for column, count in missing_counts.items()
            if count > 0
        }
        raise ValueError(f"Missing feature values detected: {details}")

    finite_mask = np.isfinite(feature_frame.to_numpy(dtype=float))
    if not finite_mask.all():
        raise ValueError("Infinite feature values are not supported.")

    if require_target:
        target_values = set(dataframe[TARGET_COLUMN].dropna().unique().tolist())
        if not target_values.issubset({0, 1}):
            raise ValueError(
                f"'{TARGET_COLUMN}' must contain only 0 and 1. Found: "
                f"{sorted(target_values)}"
            )
        if dataframe[TARGET_COLUMN].isna().any():
            raise ValueError(f"'{TARGET_COLUMN}' contains missing values.")


def split_dataset(
    dataframe: pd.DataFrame,
    *,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> DatasetSplits:
    """Create stratified train/validation/test splits.

    ``validation_size`` and ``test_size`` are expressed as fractions of the
    complete dataset.
    """
    if test_size <= 0 or validation_size <= 0:
        raise ValueError("test_size and validation_size must be greater than 0.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    X = dataframe.loc[:, FEATURE_COLUMNS].copy()
    y = dataframe[TARGET_COLUMN].astype(int).copy()

    temporary_size = test_size + validation_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temporary_size,
        stratify=y,
        random_state=random_state,
    )

    relative_test_size = test_size / temporary_size
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_state,
    )

    return DatasetSplits(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
    )


def fit_and_transform(
    splits: DatasetSplits,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a StandardScaler on training data only and transform all splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(splits.X_train)
    X_valid_scaled = scaler.transform(splits.X_valid)
    X_test_scaled = scaler.transform(splits.X_test)
    return scaler, X_train_scaled, X_valid_scaled, X_test_scaled


def save_scaler(scaler: StandardScaler, output_path: str | Path) -> Path:
    """Serialize a fitted scaler."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(scaler, file)
    return path


def save_feature_schema(output_path: str | Path) -> Path:
    """Save the production feature contract used by training and inference."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "target_column": TARGET_COLUMN,
        "required_feature_count": len(FEATURE_COLUMNS),
        "required_features": FEATURE_COLUMNS,
        "optional_columns": [TARGET_COLUMN, "transaction_id"],
    }
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return path


def create_demo_sample(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    legitimate_rows: int = 40,
    fraud_rows: int = 10,
    random_state: int = RANDOM_STATE,
) -> Path:
    """Create a deliberately mixed sample for demonstrating the application.

    The resulting sample is not representative of real-world fraud prevalence.
    It contains additional fraud rows so reviewers can see both output classes.
    """
    validate_dataset(dataframe, require_target=True)
    legitimate = dataframe[dataframe[TARGET_COLUMN] == 0]
    fraud = dataframe[dataframe[TARGET_COLUMN] == 1]

    if legitimate_rows > len(legitimate) or fraud_rows > len(fraud):
        raise ValueError("Requested sample size exceeds available class rows.")

    sample = pd.concat(
        [
            legitimate.sample(legitimate_rows, random_state=random_state),
            fraud.sample(fraud_rows, random_state=random_state),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to creditcard.csv")
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("data/sample_input.csv"),
        help="Output location for a mixed demonstration sample.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    source = load_dataset(arguments.dataset)
    created = create_demo_sample(source, arguments.sample_output)
    print(f"Demo sample saved to: {created.resolve()}")
