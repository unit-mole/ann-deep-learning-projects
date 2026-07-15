"""Synthetic retail data generator used to make the portfolio project reproducible."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import CATEGORIES, CHANNELS, RANDOM_SEED, REGIONS

CATEGORY_CONFIG = {
    "electronics": {"reference_price": 220.0, "cost_ratio": 0.72, "demand_multiplier": 1.15, "elasticity": -0.75},
    "fashion": {"reference_price": 80.0, "cost_ratio": 0.48, "demand_multiplier": 0.96, "elasticity": -1.20},
    "home": {"reference_price": 140.0, "cost_ratio": 0.58, "demand_multiplier": 1.02, "elasticity": -0.95},
    "beauty": {"reference_price": 45.0, "cost_ratio": 0.40, "demand_multiplier": 0.90, "elasticity": -1.45},
    "sports": {"reference_price": 110.0, "cost_ratio": 0.60, "demand_multiplier": 1.00, "elasticity": -1.05},
    "grocery": {"reference_price": 25.0, "cost_ratio": 0.78, "demand_multiplier": 0.78, "elasticity": -1.65},
}

CHANNEL_EFFECT = {"online": 1.04, "store": 0.98, "marketplace": 0.94}
REGION_EFFECT = {"north": 1.02, "south": 0.96, "east": 0.99, "west": 1.04}


def generate_synthetic_pricing_data(n_samples: int = 16_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate reproducible historical price-demand observations.

    The data intentionally contains realistic noise and a small amount of missingness.
    The target is observed demand at the historical/current price, not an engineered
    optimal price. This permits leakage-safe price scenario simulation at inference.
    """

    rng = np.random.default_rng(seed)
    category = rng.choice(CATEGORIES, size=n_samples, p=[0.14, 0.17, 0.20, 0.20, 0.16, 0.13])
    region = rng.choice(REGIONS, size=n_samples)
    channel = rng.choice(CHANNELS, size=n_samples, p=[0.20, 0.55, 0.25])

    day_of_year = rng.integers(1, 366, size=n_samples)
    month = np.clip(((day_of_year - 1) // 30) + 1, 1, 12)
    weekend_flag = rng.binomial(1, 0.28, size=n_samples)
    holiday_flag = rng.binomial(1, 0.08, size=n_samples)
    promotion_flag = rng.binomial(1, 0.30, size=n_samples)
    competitor_promo_flag = rng.binomial(1, 0.22, size=n_samples)

    base_cost = np.empty(n_samples)
    competitor_price = np.empty(n_samples)
    current_price = np.empty(n_samples)
    historical_sales = np.empty(n_samples)
    category_elasticity = np.empty(n_samples)
    category_demand_multiplier = np.empty(n_samples)

    for idx, cat in enumerate(category):
        cfg = CATEGORY_CONFIG[cat]
        reference = cfg["reference_price"]
        base_cost[idx] = max(rng.normal(reference * cfg["cost_ratio"], reference * 0.06), reference * 0.15)
        competitor_price[idx] = max(rng.normal(reference * 1.02, reference * 0.12), base_cost[idx] * 1.03)
        current_price[idx] = max(
            competitor_price[idx] * rng.normal(1.0, 0.10) * (0.94 if promotion_flag[idx] else 1.0),
            base_cost[idx] * 1.03,
        )
        historical_sales[idx] = max(rng.normal(95 * cfg["demand_multiplier"], 22), 5)
        category_elasticity[idx] = cfg["elasticity"]
        category_demand_multiplier[idx] = cfg["demand_multiplier"]

    rating = np.clip(rng.normal(4.15, 0.45, size=n_samples), 2.5, 5.0)
    inventory_level = np.clip(rng.normal(350, 120, size=n_samples), 30, 1_000)
    marketing_index = np.clip(rng.normal(55, 18, size=n_samples), 0, 100)
    demand_index = np.clip(rng.normal(60, 16, size=n_samples), 10, 100)
    customer_income_index = np.clip(rng.normal(100, 15, size=n_samples), 60, 160)

    seasonal_wave = np.sin(2 * np.pi * day_of_year / 365)
    peak_season = np.isin(month, [11, 12, 1]).astype(int)
    summer = np.isin(month, [6, 7, 8]).astype(int)

    baseline_demand = (
        12
        + 0.46 * historical_sales
        + 0.42 * demand_index
        + 0.15 * marketing_index
        + 8.0 * promotion_flag
        - 5.5 * competitor_promo_flag
        + 6.0 * peak_season
        + 2.5 * weekend_flag
        + 2.0 * holiday_flag
        + 4.5 * (rating - 4.0)
        + 0.08 * (customer_income_index - 100)
        + 7.0 * seasonal_wave
    )
    baseline_demand *= category_demand_multiplier
    baseline_demand *= np.array([CHANNEL_EFFECT[value] for value in channel])
    baseline_demand *= np.array([REGION_EFFECT[value] for value in region])

    relative_price = current_price / np.maximum(competitor_price, 1.0)
    realized_demand = baseline_demand * np.power(relative_price, category_elasticity)
    realized_demand *= 1 + 0.07 * promotion_flag - 0.04 * competitor_promo_flag
    realized_demand += rng.normal(0, 4.5, size=n_samples)
    realized_demand = np.clip(realized_demand, 1, None)

    # Inventory can constrain realized sales, but does not define the target directly.
    realized_demand = np.minimum(realized_demand, np.maximum(inventory_level * 0.55, 1))

    frame = pd.DataFrame(
        {
            "product_id": [f"SKU-{idx + 1:05d}" for idx in range(n_samples)],
            "day_of_year": day_of_year,
            "weekend_flag": weekend_flag,
            "holiday_flag": holiday_flag,
            "category": category,
            "region": region,
            "channel": channel,
            "promotion_flag": promotion_flag,
            "competitor_promo_flag": competitor_promo_flag,
            "base_cost": base_cost,
            "current_price": current_price,
            "competitor_price": competitor_price,
            "rating": rating,
            "inventory_level": inventory_level,
            "marketing_index": marketing_index,
            "demand_index": demand_index,
            "customer_income_index": customer_income_index,
            "historical_sales": historical_sales,
            "realized_demand": realized_demand,
        }
    )
    frame["revenue"] = frame["current_price"] * frame["realized_demand"]
    frame["gross_margin"] = (frame["current_price"] - frame["base_cost"]) * frame["realized_demand"]

    # Add low-level missingness so preprocessing is exercised.
    for column in ["rating", "marketing_index", "customer_income_index", "historical_sales"]:
        missing_mask = rng.random(n_samples) < 0.006
        frame.loc[missing_mask, column] = np.nan

    return frame
