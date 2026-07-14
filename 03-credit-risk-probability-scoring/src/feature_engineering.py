from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, RAW_REQUIRED_COLUMNS


def add_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create model-derived fields while preserving any valid supplied values."""
    result = data.copy()
    annual_income = pd.to_numeric(result.get("annual_income"), errors="coerce")
    loan_amount = pd.to_numeric(result.get("loan_amount"), errors="coerce")
    collateral_value = pd.to_numeric(result.get("collateral_value"), errors="coerce")

    calculated = {
        "monthly_income": annual_income / 12.0,
        "loan_to_income": loan_amount / annual_income.clip(lower=1.0),
        "collateral_ratio": collateral_value / loan_amount.clip(lower=1.0),
    }
    for column, values in calculated.items():
        if column not in result.columns:
            result[column] = values
        else:
            supplied = pd.to_numeric(result[column], errors="coerce")
            result[column] = supplied.where(supplied.notna(), values)
    return result


def validate_input_columns(data: pd.DataFrame) -> None:
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "Missing required input columns: " + ", ".join(missing)
            + ". Derived columns may be omitted because the pipeline creates them automatically."
        )


def prepare_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, engineer and order all fields expected by the ANN."""
    validate_input_columns(data)
    result = add_derived_features(data)
    for column in FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    return result[FEATURE_COLUMNS].copy()
