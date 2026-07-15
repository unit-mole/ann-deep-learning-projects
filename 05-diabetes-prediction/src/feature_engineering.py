"""Feature schema validation and conservative input preparation.

No speculative clinical features are created. The module preserves the eight
source indicators, converts them to numeric values, and converts implausible
zero markers to missing values for downstream train-only imputation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, ZERO_AS_MISSING_COLUMNS


def validate_feature_columns(data: pd.DataFrame) -> None:
    """Raise a clear error when required model inputs are absent."""
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + ". Download data/sample_input.csv for the expected schema."
        )


def prepare_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return model features in the correct order with safe numeric coercion."""
    validate_feature_columns(data)
    features = data.loc[:, FEATURE_COLUMNS].copy()

    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    features[ZERO_AS_MISSING_COLUMNS] = features[ZERO_AS_MISSING_COLUMNS].replace(
        0, np.nan
    )
    return features


def invalid_zero_counts(data: pd.DataFrame) -> dict[str, int]:
    """Count zero-as-missing values in the raw input frame."""
    validate_feature_columns(data)
    return {
        column: int((pd.to_numeric(data[column], errors="coerce") == 0).sum())
        for column in ZERO_AS_MISSING_COLUMNS
    }
