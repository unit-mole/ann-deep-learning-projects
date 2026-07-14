import pandas as pd

from src.risk_scoring import add_risk_outputs, assign_risk_category


def test_risk_categories():
    assert assign_risk_category(0.10) == "Low Risk"
    assert assign_risk_category(0.30) == "Medium Risk"
    assert assign_risk_category(0.80) == "High Risk"


def test_scored_output_columns():
    scored = add_risk_outputs(pd.DataFrame({"applicant": [1, 2]}), [0.10, 0.75])
    assert scored["risk_category"].tolist() == ["Low Risk", "High Risk"]
    assert "decision_recommendation" in scored.columns
