from __future__ import annotations

import numpy as np
import pandas as pd


def risk_bucket(probability: float) -> str:
    if probability >= 0.66:
        return "High"
    if probability >= 0.33:
        return "Medium"
    return "Low"


def clv_bucket(value: float) -> str:
    if value >= 16_000:
        return "High Value"
    if value >= 11_500:
        return "Mid Value"
    return "Low Value"


def engagement_bucket(score: float) -> str:
    if score >= 67:
        return "High Engagement"
    if score >= 44:
        return "Mid Engagement"
    return "Low Engagement"


def recommendation(churn_probability: float, clv: float, engagement: float) -> str:
    risk = risk_bucket(churn_probability)
    value = clv_bucket(clv)
    engagement_level = engagement_bucket(engagement)
    if risk == "High" and value == "High Value":
        return "Immediate retention outreach: protect a high-value customer with elevated churn risk."
    if risk == "High":
        return "Prioritize retention review and investigate service, inactivity, or complaint drivers."
    if risk == "Medium" and engagement_level == "Low Engagement":
        return "Use a targeted re-engagement campaign and monitor the account closely."
    if value == "High Value" and engagement_level == "High Engagement":
        return "Maintain service quality and consider personalized cross-sell or loyalty offers."
    return "Continue standard engagement and monitor changes in risk, value, and activity."


def add_business_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["churn_risk_bucket"] = result["predicted_churn_probability"].map(risk_bucket)
    result["clv_bucket"] = result["predicted_clv"].map(clv_bucket)
    result["engagement_bucket"] = result["predicted_engagement"].map(engagement_bucket)
    result["recommended_action"] = [
        recommendation(p, c, e)
        for p, c, e in zip(
            result["predicted_churn_probability"],
            result["predicted_clv"],
            result["predicted_engagement"],
        )
    ]
    return result
