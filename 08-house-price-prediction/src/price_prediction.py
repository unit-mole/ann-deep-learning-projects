"""Business-facing formatting for ANN price predictions."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .config import TARGET_MULTIPLIER
except ImportError:
    from config import TARGET_MULTIPLIER


def target_units_to_usd(value: float | np.ndarray) -> float | np.ndarray:
    """Convert California Housing target units into US dollars."""
    return np.asarray(value) * TARGET_MULTIPLIER


def assign_price_category(
    prediction_target_units: float,
    thresholds: dict[str, float],
) -> str:
    """Assign a distribution-relative price category."""
    prediction = float(prediction_target_units)
    if prediction < thresholds["budget_upper"]:
        return "Budget Property"
    if prediction < thresholds["mid_range_upper"]:
        return "Mid-Range Property"
    if prediction < thresholds["premium_upper"]:
        return "Premium Property"
    return "Luxury Property"


def estimated_price_range_usd(
    prediction_target_units: float,
    half_width_target_units: float,
) -> tuple[float, float]:
    """Create an empirical error range around a point estimate."""
    lower = max(0.0, prediction_target_units - half_width_target_units)
    upper = prediction_target_units + half_width_target_units
    return lower * TARGET_MULTIPLIER, upper * TARGET_MULTIPLIER


def build_business_interpretation(
    category: str,
    local_drivers: Iterable[dict[str, object]] | None = None,
) -> str:
    """Create a concise explanation without overstating causal effects."""
    base = (
        f"The estimate falls in the **{category}** segment relative to the "
        "training-data price distribution."
    )
    drivers = list(local_drivers or [])
    if not drivers:
        return (
            base
            + " The ANN combines income, geographic coordinates, housing age, "
            "room structure, population, and occupancy through nonlinear relationships."
        )

    phrases = []
    for driver in drivers[:3]:
        direction = "increased" if float(driver["impact_usd"]) > 0 else "reduced"
        phrases.append(f"{driver['display_name']} {direction} the estimate")
    return base + " The strongest local signals were: " + "; ".join(phrases) + "."


def append_business_outputs(
    frame: pd.DataFrame,
    predictions_target_units: np.ndarray,
    thresholds: dict[str, float],
    interval_half_width: float,
) -> pd.DataFrame:
    """Attach dollar values, empirical ranges, and categories to batch predictions."""
    output = frame.copy()
    prediction_array = np.asarray(predictions_target_units, dtype=float).ravel()
    output["PredictedSalePrice"] = prediction_array
    output["PredictedPriceUSD"] = prediction_array * TARGET_MULTIPLIER
    output["EstimatedLowUSD"] = np.maximum(
        0.0, prediction_array - interval_half_width
    ) * TARGET_MULTIPLIER
    output["EstimatedHighUSD"] = (
        prediction_array + interval_half_width
    ) * TARGET_MULTIPLIER
    output["PriceCategory"] = [
        assign_price_category(value, thresholds) for value in prediction_array
    ]
    return output
