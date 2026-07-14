from __future__ import annotations

import pandas as pd

from src.config import (
    ALLOWED_GENDERS,
    ALLOWED_GEOGRAPHIES,
    NUMERIC_RANGES,
    REQUIRED_FEATURES,
)


class InputValidationError(ValueError):
    """Raised when prediction input does not satisfy the project schema."""


def _coerce_binary(series: pd.Series, column: str) -> pd.Series:
    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
    }

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    converted = normalized.map(mapping)

    numeric = pd.to_numeric(series, errors="coerce")
    converted = converted.fillna(numeric)

    if converted.isna().any() or not converted.isin([0, 1]).all():
        raise InputValidationError(
            f"Column '{column}' must contain 0/1, True/False, or Yes/No values."
        )
    return converted.astype(int)


def validate_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a frame used for single or batch prediction."""
    if not isinstance(frame, pd.DataFrame):
        raise InputValidationError("Prediction input must be a pandas DataFrame.")

    missing = [column for column in REQUIRED_FEATURES if column not in frame.columns]
    if missing:
        raise InputValidationError(
            "Missing required column(s): " + ", ".join(missing)
        )

    validated = frame.loc[:, REQUIRED_FEATURES].copy()

    if validated.empty:
        raise InputValidationError("The input file contains no rows.")

    if validated.isna().any().any():
        null_columns = validated.columns[validated.isna().any()].tolist()
        raise InputValidationError(
            "Missing values found in: " + ", ".join(null_columns)
        )

    validated["Geography"] = validated["Geography"].astype(str).str.strip()
    validated["Gender"] = validated["Gender"].astype(str).str.strip().str.title()

    invalid_geo = sorted(set(validated["Geography"]) - ALLOWED_GEOGRAPHIES)
    if invalid_geo:
        raise InputValidationError(
            "Unsupported Geography value(s): " + ", ".join(invalid_geo)
        )

    invalid_gender = sorted(set(validated["Gender"]) - ALLOWED_GENDERS)
    if invalid_gender:
        raise InputValidationError(
            "Unsupported Gender value(s): " + ", ".join(invalid_gender)
        )

    validated["HasCrCard"] = _coerce_binary(validated["HasCrCard"], "HasCrCard")
    validated["IsActiveMember"] = _coerce_binary(
        validated["IsActiveMember"], "IsActiveMember"
    )

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
        if validated[column].isna().any():
            raise InputValidationError(
                f"Column '{column}' contains a non-numeric value."
            )
        invalid = ~validated[column].between(minimum, maximum, inclusive="both")
        if invalid.any():
            raise InputValidationError(
                f"Column '{column}' must be between {minimum} and {maximum}."
            )

    integer_columns = [
        "CreditScore",
        "Age",
        "Tenure",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
    ]
    validated[integer_columns] = validated[integer_columns].astype(int)

    return validated
