import pandas as pd
from src.feature_engineering import add_deployment_features


def test_feature_engineering_adds_expected_columns():
    frame = pd.DataFrame([{
        "age":35,"tenure_months":10,"monthly_income":6500,"monthly_spend":350,
        "transactions_last_30d":8,"avg_session_minutes":24,"products_owned":3,
        "complaints_last_6m":1,"days_since_last_login":14,"discount_usage_rate":.3,
        "support_tickets_last_90d":1,"website_visits_last_30d":14,"region":"West",
        "channel":"Mobile","plan_type":"Standard","segment":"Growth","device_type":"iOS",
    }])
    metadata={
        "spend_bin_edges":[-float("inf"),300,420,540,float("inf")],
        "spend_bin_labels":["Low","Mid-Low","Mid-High","High"],
        "tenure_bin_edges":[-float("inf"),6,12,24,48,float("inf")],
        "tenure_bin_labels":["0-6","7-12","13-24","25-48","49+"],
    }
    result=add_deployment_features(frame,metadata)
    assert result.loc[0,"tenure_cohort"]=="7-12"
    assert result.loc[0,"spend_band"]=="Mid-Low"
    assert "customer_segment_cluster" not in result.columns
