"""Feature schema validation and conservative input preparation.

No speculative clinical features are created. The module preserves the eight
source indicators, validates uploaded values, converts them to numeric form,
and converts documented zero markers to missing values for downstream
train-only imputation.
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
            "Missing required columns: "
            + ", ".join(missing)
            + ". Download data/sample_input.csv for the expected schema."
        )


def _row_labels(mask: pd.Series, limit: int = 5) -> str:
    """Return compact, user-facing row labels for validation messages."""
    labels = [str(index) for index in mask.index[mask].tolist()[:limit]]
    suffix = "..." if int(mask.sum()) > limit else ""
    return ", ".join(labels) + suffix


def validate_feature_values(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and convert feature values without silently coercing bad text.

    Explicit missing values are allowed because the saved preprocessing pipeline
    performs median imputation. Unexpected text, infinite values, and negative
    measurements are rejected with a clear message before prediction.
    """
    validate_feature_columns(data)

    if data.empty:
        raise ValueError("The input file contains no data rows to score.")

    numeric = data.loc[:, FEATURE_COLUMNS].copy()
    errors: list[str] = []

    for column in FEATURE_COLUMNS:
        raw = numeric[column]
        converted = pd.to_numeric(raw, errors="coerce")

        invalid_text = raw.notna() & converted.isna()
        if invalid_text.any():
            examples = raw.loc[invalid_text].astype(str).drop_duplicates().tolist()[:3]
            errors.append(
                f"{column}: non-numeric value(s) {examples} at row index(es) "
                f"{_row_labels(invalid_text)}"
            )

        non_finite = converted.notna() & ~np.isfinite(converted)
        if non_finite.any():
            errors.append(
                f"{column}: infinite value(s) at row index(es) "
                f"{_row_labels(non_finite)}"
            )

        negative = converted.notna() & (converted < 0)
        if negative.any():
            errors.append(
                f"{column}: negative value(s) at row index(es) "
                f"{_row_labels(negative)}"
            )

        numeric[column] = converted

    if errors:
        raise ValueError(
            "Invalid input data. Correct the following issue(s):\n- "
            + "\n- ".join(errors)
        )

    return numeric


def prepare_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return validated model features with documented zero markers as missing."""
    features = validate_feature_values(data)
    features[ZERO_AS_MISSING_COLUMNS] = features[ZERO_AS_MISSING_COLUMNS].replace(
        0, np.nan
    )
    return features


def invalid_zero_counts(data: pd.DataFrame) -> dict[str, int]:
    """Count zero-as-missing values in the raw input frame."""
    numeric = validate_feature_values(data)
    return {
        column: int((numeric[column] == 0).sum())
        for column in ZERO_AS_MISSING_COLUMNS
    }
