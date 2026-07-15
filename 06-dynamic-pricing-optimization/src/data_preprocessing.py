"""Data validation and cleaning functions."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .constants import CATEGORIES, CHANNELS, DEFAULT_RECORD, REGIONS, REQUIRED_INPUT_COLUMNS

NUMERIC_DEFAULTS = {
    key: value
    for key, value in DEFAULT_RECORD.items()
    if isinstance(value, (int, float)) and key not in {"product_id"}
}


def validate_input_columns(frame: pd.DataFrame, required: Iterable[str] = REQUIRED_INPUT_COLUMNS) -> list[str]:
    """Return required columns that are absent from a frame."""

    return [column for column in required if column not in frame.columns]


def clean_pricing_data(frame: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    """Clean user or training data while preserving the original row order."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned = frame.copy()
    missing = validate_input_columns(cleaned)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if require_target and "realized_demand" not in cleaned.columns:
        raise ValueError("Training data must contain 'realized_demand'.")

    if "product_id" not in cleaned.columns:
        cleaned.insert(0, "product_id", [f"ROW-{idx + 1:05d}" for idx in range(len(cleaned))])
    cleaned["product_id"] = cleaned["product_id"].fillna("").astype(str)

    for column, default in NUMERIC_DEFAULTS.items():
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        median = cleaned[column].median(skipna=True)
        fill_value = float(median) if np.isfinite(median) else float(default)
        cleaned[column] = cleaned[column].fillna(fill_value)

    if require_target:
        cleaned["realized_demand"] = pd.to_numeric(cleaned["realized_demand"], errors="coerce")
        cleaned = cleaned.dropna(subset=["realized_demand"]).copy()
        cleaned["realized_demand"] = cleaned["realized_demand"].clip(lower=0.1)

    cleaned["category"] = cleaned["category"].fillna(DEFAULT_RECORD["category"]).astype(str).str.lower().str.strip()
    cleaned["region"] = cleaned["region"].fillna(DEFAULT_RECORD["region"]).astype(str).str.lower().str.strip()
    cleaned["channel"] = cleaned["channel"].fillna(DEFAULT_RECORD["channel"]).astype(str).str.lower().str.strip()

    cleaned.loc[~cleaned["category"].isin(CATEGORIES), "category"] = DEFAULT_RECORD["category"]
    cleaned.loc[~cleaned["region"].isin(REGIONS), "region"] = DEFAULT_RECORD["region"]
    cleaned.loc[~cleaned["channel"].isin(CHANNELS), "channel"] = DEFAULT_RECORD["channel"]

    cleaned["day_of_year"] = cleaned["day_of_year"].round().clip(1, 365).astype(int)
    for flag in ["weekend_flag", "holiday_flag", "promotion_flag", "competitor_promo_flag"]:
        cleaned[flag] = cleaned[flag].round().clip(0, 1).astype(int)

    cleaned["base_cost"] = cleaned["base_cost"].clip(lower=0.01)
    cleaned["current_price"] = cleaned["current_price"].clip(lower=0.01)
    cleaned["competitor_price"] = cleaned["competitor_price"].clip(lower=0.01)
    cleaned["rating"] = cleaned["rating"].clip(1.0, 5.0)
    cleaned["inventory_level"] = cleaned["inventory_level"].clip(lower=0.0)
    cleaned["marketing_index"] = cleaned["marketing_index"].clip(0.0, 100.0)
    cleaned["demand_index"] = cleaned["demand_index"].clip(0.0, 100.0)
    cleaned["customer_income_index"] = cleaned["customer_income_index"].clip(1.0, 250.0)
    cleaned["historical_sales"] = cleaned["historical_sales"].clip(lower=0.0)

    return cleaned.reset_index(drop=True)
