from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .embedding_preprocessing import fit_category_encoders, transform_category_frame


@dataclass
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    frame: pd.DataFrame,
    target_column: str,
    seed: int = 42,
) -> DataSplits:
    """Create stratified 64/16/20 train/validation/test partitions."""
    train, test = train_test_split(
        frame,
        test_size=0.20,
        random_state=seed,
        stratify=frame[target_column],
    )
    train, validation = train_test_split(
        train,
        test_size=0.20,
        random_state=seed,
        stratify=train[target_column],
    )
    return DataSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def fit_numeric_preprocessors(
    train_frame: pd.DataFrame,
    numerical_features: list[str],
) -> tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(train_frame[numerical_features])
    scaler.fit(imputed)
    return imputer, scaler


def build_model_inputs(
    frame: pd.DataFrame,
    categorical_features: list[str],
    numerical_features: list[str],
    category_encoders: dict[str, object],
    numeric_imputer: SimpleImputer,
    numeric_scaler: StandardScaler,
    fallback_categories: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    numeric = frame[numerical_features].apply(pd.to_numeric, errors="coerce")
    numeric_array = numeric_imputer.transform(numeric)
    numeric_array = numeric_scaler.transform(numeric_array).astype("float32")

    categorical_arrays, unknown_counts = transform_category_frame(
        frame,
        category_encoders,
        categorical_features,
        fallback_categories=fallback_categories,
    )
    categorical_arrays["numerical_input"] = numeric_array
    return categorical_arrays, unknown_counts


def fit_all_preprocessors(
    train_frame: pd.DataFrame,
    categorical_features: list[str],
    numerical_features: list[str],
):
    category_encoders = fit_category_encoders(train_frame, categorical_features)
    numeric_imputer, numeric_scaler = fit_numeric_preprocessors(train_frame, numerical_features)
    return category_encoders, numeric_imputer, numeric_scaler
