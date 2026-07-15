import pandas as pd

from src.feature_engineering import add_derived_features, infer_legacy_customer_segment


def test_derived_rate_features() -> None:
    frame = pd.DataFrame({"n_orders": [12], "total_revenue": [1200.0], "total_quantity": [48], "tenure_days": [360]})
    result = add_derived_features(frame)
    assert result.loc[0, "avg_order_value"] == 100.0
    assert result.loc[0, "avg_quantity"] == 4.0
    assert result.loc[0, "orders_per_month"] == 1.0
    assert result.loc[0, "revenue_per_month"] == 100.0


def test_segment_fallback() -> None:
    frame = pd.DataFrame({"n_orders": [15, 2], "total_revenue": [1800.0, 80.0], "recency_days": [20, 220]})
    result = infer_legacy_customer_segment(frame)
    assert result.iloc[0] == "Segment_2"
    assert result.iloc[1] == "Segment_1"
