from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CLASSIFICATION_THRESHOLD, LOW_RISK_UPPER, MEDIUM_RISK_UPPER


@dataclass(frozen=True)
class RiskDecision:
    category: str
    recommendation: str
    review_priority: str


def assign_risk_category(
    probability: float,
    low_upper: float = LOW_RISK_UPPER,
    medium_upper: float = MEDIUM_RISK_UPPER,
) -> str:
    probability = float(np.clip(probability, 0.0, 1.0))
    if probability < low_upper:
        return "Low Risk"
    if probability < medium_upper:
        return "Medium Risk"
    return "High Risk"


def decision_for_probability(probability: float) -> RiskDecision:
    category = assign_risk_category(probability)
    if category == "Low Risk":
        return RiskDecision(category, "Approve / standard underwriting checks", "Routine")
    if category == "Medium Risk":
        return RiskDecision(category, "Manual review / verify supporting documents", "Elevated")
    return RiskDecision(category, "Decline or apply enhanced credit scrutiny", "High")


def add_risk_outputs(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    classification_threshold: float = CLASSIFICATION_THRESHOLD,
) -> pd.DataFrame:
    result = data.reset_index(drop=True).copy()
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(result) != len(probs):
        raise ValueError("Input rows and probabilities must have the same length.")
    result["risk_probability"] = probs
    result["risk_probability_percent"] = probs * 100.0
    result["predicted_default"] = (probs >= classification_threshold).astype(int)
    decisions = [decision_for_probability(value) for value in probs]
    result["risk_category"] = [item.category for item in decisions]
    result["decision_recommendation"] = [item.recommendation for item in decisions]
    result["review_priority"] = [item.review_priority for item in decisions]
    result["classification_threshold"] = classification_threshold
    return result
