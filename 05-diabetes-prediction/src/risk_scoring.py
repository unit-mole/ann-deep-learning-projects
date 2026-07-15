"""Risk banding and user-facing portfolio interpretations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_RISK_THRESHOLDS

RISK_RECOMMENDATIONS = {
    "Low Risk": (
        "The model estimates a lower relative risk for this input. Maintain routine "
        "health monitoring and discuss concerns with a qualified professional."
    ),
    "Medium Risk": (
        "The model estimates an elevated screening risk. Consider discussing the "
        "health indicators with a qualified healthcare professional."
    ),
    "High Risk": (
        "The model estimates a higher screening risk. Consult a qualified healthcare "
        "professional for appropriate evaluation; this is not a diagnosis."
    ),
}


def probability_to_risk_category(
    probability: float,
    low_upper: float = DEFAULT_RISK_THRESHOLDS["low_upper"],
    high_lower: float = DEFAULT_RISK_THRESHOLDS["high_lower"],
) -> str:
    """Map one probability to a transparent communication band."""
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Probability must be between 0 and 1.")
    if not 0.0 < low_upper < high_lower < 1.0:
        raise ValueError("Risk thresholds must satisfy 0 < low < high < 1.")
    if value < low_upper:
        return "Low Risk"
    if value < high_lower:
        return "Medium Risk"
    return "High Risk"


def categorize_probabilities(probabilities: np.ndarray | pd.Series) -> pd.Series:
    """Vectorized risk category assignment."""
    values = np.asarray(probabilities, dtype=float)
    return pd.Series(
        np.select(
            [values < DEFAULT_RISK_THRESHOLDS["low_upper"],
             values < DEFAULT_RISK_THRESHOLDS["high_lower"]],
            ["Low Risk", "Medium Risk"],
            default="High Risk",
        )
    )


def recommendation_for_category(category: str) -> str:
    """Return a cautious, non-diagnostic interpretation."""
    try:
        return RISK_RECOMMENDATIONS[category]
    except KeyError as exc:
        raise ValueError(f"Unknown risk category: {category}") from exc
