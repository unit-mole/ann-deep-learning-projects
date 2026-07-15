"""Price scenario generation and objective scoring."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .pricing_rules import calculate_price_bounds

DemandPredictor = Callable[[pd.DataFrame], np.ndarray]


def _normalize(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    if np.isclose(maximum, minimum):
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return (values - minimum) / (maximum - minimum)


def build_candidate_prices(
    *,
    current_price: float,
    base_cost: float,
    competitor_price: float,
    inventory_level: float,
    baseline_demand: float,
    n_grid: int = 31,
) -> tuple[np.ndarray, tuple[str, ...]]:
    bounds = calculate_price_bounds(
        current_price=current_price,
        base_cost=base_cost,
        competitor_price=competitor_price,
        inventory_level=inventory_level,
        baseline_demand=baseline_demand,
    )
    candidates = np.linspace(bounds.minimum, bounds.maximum, max(int(n_grid), 5))
    candidates = np.unique(np.append(candidates, current_price))
    return np.sort(candidates), bounds.notes


def optimize_price_scenarios(
    record: pd.Series,
    demand_predictor: DemandPredictor,
    objective: str = "margin",
    n_grid: int = 31,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Predict demand across candidate prices and select the best objective score."""

    baseline_frame = pd.DataFrame([record.to_dict()])
    baseline_demand = float(demand_predictor(baseline_frame)[0])
    candidates, notes = build_candidate_prices(
        current_price=float(record["current_price"]),
        base_cost=float(record["base_cost"]),
        competitor_price=float(record["competitor_price"]),
        inventory_level=float(record["inventory_level"]),
        baseline_demand=baseline_demand,
        n_grid=n_grid,
    )

    scenarios = pd.DataFrame([record.to_dict()] * len(candidates))
    scenarios["current_price"] = candidates
    scenarios["predicted_demand"] = demand_predictor(scenarios)
    scenarios["predicted_revenue"] = scenarios["current_price"] * scenarios["predicted_demand"]
    scenarios["predicted_margin"] = (
        scenarios["current_price"] - scenarios["base_cost"]
    ) * scenarios["predicted_demand"]

    if objective == "revenue":
        scenarios["objective_score"] = scenarios["predicted_revenue"]
    elif objective == "balanced":
        scenarios["objective_score"] = (
            0.60 * _normalize(scenarios["predicted_margin"])
            + 0.25 * _normalize(scenarios["predicted_revenue"])
            + 0.15 * _normalize(scenarios["predicted_demand"])
        )
    else:
        scenarios["objective_score"] = scenarios["predicted_margin"]

    best_index = scenarios["objective_score"].idxmax()
    scenarios["selected"] = False
    scenarios.loc[best_index, "selected"] = True
    scenarios["price_change_pct"] = (
        scenarios["current_price"] - float(record["current_price"])
    ) / max(float(record["current_price"]), 1e-6)

    return scenarios.reset_index(drop=True), notes
