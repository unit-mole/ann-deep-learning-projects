import pandas as pd

from src.constants import DEFAULT_RECORD, NUMERIC_FEATURE_COLUMNS
from src.data_preprocessing import clean_pricing_data
from src.feature_engineering import add_pricing_features


def test_feature_engineering_creates_complete_numeric_schema():
    frame = pd.DataFrame([DEFAULT_RECORD])
    featured = add_pricing_features(clean_pricing_data(frame))
    assert set(NUMERIC_FEATURE_COLUMNS).issubset(featured.columns)
    assert featured[NUMERIC_FEATURE_COLUMNS].isna().sum().sum() == 0
    assert featured.loc[0, "price_gap_vs_competitor_pct"] < 0
