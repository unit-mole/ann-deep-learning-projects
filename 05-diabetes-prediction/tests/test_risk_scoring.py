import pytest

from src.risk_scoring import probability_to_risk_category


def test_risk_boundaries():
    assert probability_to_risk_category(0.0) == "Low Risk"
    assert probability_to_risk_category(0.2999) == "Low Risk"
    assert probability_to_risk_category(0.30) == "Medium Risk"
    assert probability_to_risk_category(0.5999) == "Medium Risk"
    assert probability_to_risk_category(0.60) == "High Risk"
    assert probability_to_risk_category(1.0) == "High Risk"


def test_invalid_probability():
    with pytest.raises(ValueError):
        probability_to_risk_category(1.1)
