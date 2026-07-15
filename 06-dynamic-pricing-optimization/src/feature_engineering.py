"""Leakage-safe feature engineering for training and live inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import NUMERIC_FEATURE_COLUMNS


def add_pricing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, price, margin, and inventory features available at decision time."""

    featured = frame.copy()
    featured["month"] = np.clip(((featured["day_of_year"] - 1) // 30) + 1, 1, 12).astype(int)
    featured["week_of_year"] = np.clip(((featured["day_of_year"] - 1) // 7) + 1, 1, 52).astype(int)
    featured["peak_season_flag"] = featured["month"].isin([11, 12, 1]).astype(int)
    featured["summer_flag"] = featured["month"].isin([6, 7, 8]).astype(int)

    featured["seasonal_wave_1"] = np.sin(2 * np.pi * featured["day_of_year"] / 365)
    featured["seasonal_wave_2"] = np.cos(2 * np.pi * featured["day_of_year"] / 365)
    featured["month_sin"] = np.sin(2 * np.pi * featured["month"] / 12)
    featured["month_cos"] = np.cos(2 * np.pi * featured["month"] / 12)
    featured["week_sin"] = np.sin(2 * np.pi * featured["week_of_year"] / 52)
    featured["week_cos"] = np.cos(2 * np.pi * featured["week_of_year"] / 52)

    competitor_denominator = np.maximum(featured["competitor_price"], 1e-6)
    cost_denominator = np.maximum(featured["base_cost"], 1e-6)
    sales_denominator = np.maximum(featured["historical_sales"], 1.0)

    featured["price_gap_vs_competitor_pct"] = (
        featured["current_price"] - featured["competitor_price"]
    ) / competitor_denominator
    featured["markup_pct"] = (featured["current_price"] - featured["base_cost"]) / cost_denominator
    featured["inventory_pressure"] = featured["inventory_level"] / sales_denominator
    featured["promo_competition_interaction"] = (
        featured["promotion_flag"] * featured["competitor_promo_flag"]
    )

    non_finite_mask = ~np.isfinite(featured[NUMERIC_FEATURE_COLUMNS])
    if non_finite_mask.any().any():
        featured[NUMERIC_FEATURE_COLUMNS] = featured[NUMERIC_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        featured[NUMERIC_FEATURE_COLUMNS] = featured[NUMERIC_FEATURE_COLUMNS].fillna(0.0)

    return featured
