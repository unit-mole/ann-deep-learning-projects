"""Shared schema and business constants for the dynamic pricing project."""

from __future__ import annotations

CATEGORIES = ["beauty", "electronics", "fashion", "grocery", "home", "sports"]
REGIONS = ["east", "north", "south", "west"]
CHANNELS = ["marketplace", "online", "store"]

RAW_INPUT_COLUMNS = [
    "product_id",
    "day_of_year",
    "weekend_flag",
    "holiday_flag",
    "category",
    "region",
    "channel",
    "promotion_flag",
    "competitor_promo_flag",
    "base_cost",
    "current_price",
    "competitor_price",
    "rating",
    "inventory_level",
    "marketing_index",
    "demand_index",
    "customer_income_index",
    "historical_sales",
]

REQUIRED_INPUT_COLUMNS = [column for column in RAW_INPUT_COLUMNS if column != "product_id"]

NUMERIC_FEATURE_COLUMNS = [
    "day_of_year",
    "month",
    "week_of_year",
    "weekend_flag",
    "holiday_flag",
    "peak_season_flag",
    "summer_flag",
    "promotion_flag",
    "competitor_promo_flag",
    "base_cost",
    "current_price",
    "competitor_price",
    "rating",
    "inventory_level",
    "marketing_index",
    "demand_index",
    "customer_income_index",
    "historical_sales",
    "seasonal_wave_1",
    "seasonal_wave_2",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "price_gap_vs_competitor_pct",
    "markup_pct",
    "inventory_pressure",
    "promo_competition_interaction",
]

CATEGORICAL_COLUMNS = ["category", "region", "channel"]
TARGET_COLUMN = "realized_demand"
CLASSIFICATION_TARGET_COLUMN = "high_demand_flag"

SEGMENT_FEATURE_COLUMNS = [
    "current_price",
    "price_gap_vs_competitor_pct",
    "inventory_pressure",
    "demand_index",
    "historical_sales",
    "predicted_demand",
]

DEFAULT_RECORD = {
    "product_id": "DEMO-001",
    "day_of_year": 320,
    "weekend_flag": 1,
    "holiday_flag": 0,
    "category": "electronics",
    "region": "west",
    "channel": "online",
    "promotion_flag": 0,
    "competitor_promo_flag": 0,
    "base_cost": 165.0,
    "current_price": 250.0,
    "competitor_price": 265.0,
    "rating": 4.6,
    "inventory_level": 180.0,
    "marketing_index": 82.0,
    "demand_index": 88.0,
    "customer_income_index": 122.0,
    "historical_sales": 128.0,
}

OBJECTIVES = {
    "Maximize Margin": "margin",
    "Maximize Revenue": "revenue",
    "Balance Demand and Margin": "balanced",
}

MODEL_VERSION = "2.0.0"
RANDOM_SEED = 42
