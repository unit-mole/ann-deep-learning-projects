from __future__ import annotations

import pandas as pd


def validate_feature_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def probability_bucket(probability: float) -> str:
    if probability >= 0.80:
        return "Very high"
    if probability >= 0.60:
        return "High"
    if probability >= 0.40:
        return "Medium"
    if probability >= 0.20:
        return "Low"
    return "Very low"


def business_interpretation(probability: float, threshold: float) -> str:
    outcome = "higher" if probability >= threshold else "lower"
    return (
        f"The model estimates {outcome} positive-outcome propensity by combining "
        "scaled numerical behavior with learned categorical embedding vectors."
    )
