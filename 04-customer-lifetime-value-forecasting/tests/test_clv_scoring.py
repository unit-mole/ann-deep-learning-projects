from src.clv_scoring import SegmentThresholds, assign_clv_segment, business_recommendation


def test_segment_boundaries() -> None:
    thresholds = SegmentThresholds(low_max=100, medium_max=200, high_max=300)
    assert assign_clv_segment(0, thresholds) == "Low Value Customer"
    assert assign_clv_segment(100, thresholds) == "Low Value Customer"
    assert assign_clv_segment(101, thresholds) == "Medium Value Customer"
    assert assign_clv_segment(250, thresholds) == "High Value Customer"
    assert assign_clv_segment(301, thresholds) == "VIP / Strategic Customer"


def test_vip_at_risk_recommendation_is_retention_focused() -> None:
    text = business_recommendation("VIP / Strategic Customer", 0.2).lower()
    assert "retention" in text
