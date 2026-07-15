"""Production-style model loading, demand inference, and price optimization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import keras
import numpy as np
import pandas as pd

from .constants import CATEGORICAL_COLUMNS, NUMERIC_FEATURE_COLUMNS, OBJECTIVES
from .data_preprocessing import clean_pricing_data
from .feature_engineering import add_pricing_features
from .price_optimization import optimize_price_scenarios
from .pricing_rules import build_business_recommendation, determine_pricing_action

SEGMENT_NAMES = {
    0: "Value / Low Velocity",
    1: "Core / Stable Demand",
    2: "Premium / High Value",
    3: "Promotion-Sensitive",
}


class DynamicPricingPipeline:
    """Load saved artifacts once and expose single-row and batch recommendations."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.regressor = keras.models.load_model(self.model_dir / "dynamic_pricing_demand_ann.keras", compile=False)
        self.classifier = keras.models.load_model(self.model_dir / "dynamic_pricing_demand_state_ann.keras", compile=False)
        self.numeric_scaler = joblib.load(self.model_dir / "numeric_scaler.joblib")
        self.metadata = json.loads((self.model_dir / "model_metadata.json").read_text(encoding="utf-8"))
        self.encoder_maps: dict[str, dict[str, int]] = self.metadata["encoder_maps"]

    def _prepare(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[np.ndarray]]:
        clean = clean_pricing_data(frame, require_target=False)
        featured = add_pricing_features(clean)
        numeric = self.numeric_scaler.transform(featured[NUMERIC_FEATURE_COLUMNS]).astype("float32")
        encoded = {
            column: featured[column].map(self.encoder_maps[column]).fillna(0).astype("int32").to_numpy()
            for column in CATEGORICAL_COLUMNS
        }
        model_inputs = [numeric, encoded["category"], encoded["region"], encoded["channel"]]
        return featured, model_inputs

    @staticmethod
    def _forward_in_batches(model: keras.Model, model_inputs: list[np.ndarray], batch_size: int = 256) -> np.ndarray:
        outputs: list[np.ndarray] = []
        n_rows = len(model_inputs[0])
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            chunk = [array[start:end] for array in model_inputs]
            outputs.append(keras.ops.convert_to_numpy(model(chunk, training=False)).reshape(-1))
        return np.concatenate(outputs) if outputs else np.array([], dtype=float)

    def predict_demand(self, frame: pd.DataFrame) -> np.ndarray:
        _, model_inputs = self._prepare(frame)
        log_prediction = self._forward_in_batches(self.regressor, model_inputs)
        return np.clip(np.expm1(log_prediction), 0.0, None)

    def predict_high_demand_probability(self, frame: pd.DataFrame) -> np.ndarray:
        _, model_inputs = self._prepare(frame)
        return self._forward_in_batches(self.classifier, model_inputs)

    def assign_segment(self, featured_row: pd.Series, predicted_demand: float) -> tuple[int, str]:
        """Assign a transparent, inference-safe pricing segment."""

        high_cutoff = float(self.metadata["high_demand_cutoff_units"])
        current_price = float(featured_row["current_price"])
        competitor_price = float(featured_row["competitor_price"])
        inventory_pressure = float(featured_row["inventory_pressure"])
        promotion_flag = int(featured_row["promotion_flag"])
        price_gap = (current_price - competitor_price) / max(competitor_price, 1e-6)

        if current_price >= competitor_price * 1.08 and predicted_demand >= high_cutoff:
            segment_id = 2
        elif promotion_flag == 1 or inventory_pressure > 4.0 or (price_gap > 0.10 and predicted_demand < high_cutoff):
            segment_id = 3
        elif predicted_demand >= high_cutoff * 0.80 and 1.0 <= inventory_pressure <= 4.0:
            segment_id = 1
        else:
            segment_id = 0
        return segment_id, SEGMENT_NAMES[segment_id]

    def recommend_one(
        self,
        record: dict[str, Any] | pd.Series,
        objective_label: str = "Maximize Margin",
        n_grid: int = 31,
    ) -> dict[str, Any]:
        objective = OBJECTIVES.get(objective_label, objective_label)
        raw_frame = pd.DataFrame([record.to_dict() if isinstance(record, pd.Series) else record])
        clean = clean_pricing_data(raw_frame, require_target=False)
        featured = add_pricing_features(clean)
        row = featured.iloc[0]

        current_demand = float(self.predict_demand(clean)[0])
        current_revenue = float(row["current_price"] * current_demand)
        current_margin = float((row["current_price"] - row["base_cost"]) * current_demand)
        high_demand_probability = float(self.predict_high_demand_probability(clean)[0])

        scenarios, rule_notes = optimize_price_scenarios(
            row,
            demand_predictor=self.predict_demand,
            objective=objective,
            n_grid=n_grid,
        )
        selected = scenarios.loc[scenarios["selected"]].iloc[0]
        recommended_price = float(selected["current_price"])
        optimized_demand = float(selected["predicted_demand"])
        optimized_revenue = float(selected["predicted_revenue"])
        optimized_margin = float(selected["predicted_margin"])
        change_pct = (recommended_price - float(row["current_price"])) / max(float(row["current_price"]), 1e-6)

        action = determine_pricing_action(
            current_price=float(row["current_price"]),
            recommended_price=recommended_price,
            inventory_level=float(row["inventory_level"]),
            baseline_demand=current_demand,
            competitor_price=float(row["competitor_price"]),
        )
        recommendation = build_business_recommendation(
            action=action,
            change_pct=change_pct,
            inventory_level=float(row["inventory_level"]),
            optimized_demand=optimized_demand,
            competitor_price=float(row["competitor_price"]),
            recommended_price=recommended_price,
            objective_label=objective_label,
            rule_notes=rule_notes,
        )
        segment_id, segment_name = self.assign_segment(row, current_demand)

        return {
            "product_id": row.get("product_id", ""),
            "current_price": float(row["current_price"]),
            "recommended_price": recommended_price,
            "price_change_pct": change_pct,
            "predicted_demand_current": current_demand,
            "predicted_demand_optimized": optimized_demand,
            "predicted_revenue_current": current_revenue,
            "predicted_revenue_optimized": optimized_revenue,
            "predicted_margin_current": current_margin,
            "predicted_margin_optimized": optimized_margin,
            "estimated_margin_uplift": optimized_margin - current_margin,
            "high_demand_probability": high_demand_probability,
            "pricing_segment_id": segment_id,
            "pricing_segment": segment_name,
            "pricing_action": action,
            "business_recommendation": recommendation,
            "objective": objective_label,
            "scenario_table": scenarios,
            "rule_notes": rule_notes,
        }

    def recommend_batch(
        self,
        frame: pd.DataFrame,
        objective_label: str = "Maximize Margin",
        n_grid: int = 17,
    ) -> pd.DataFrame:
        """Optimize a batch with vectorized ANN inference across all price scenarios."""

        from .pricing_rules import calculate_price_bounds

        objective = OBJECTIVES.get(objective_label, objective_label)
        clean = clean_pricing_data(frame, require_target=False)
        featured = add_pricing_features(clean)
        baseline_demand = self.predict_demand(clean)
        high_probabilities = self.predict_high_demand_probability(clean)

        scenario_frames: list[pd.DataFrame] = []
        rule_notes_by_row: dict[int, tuple[str, ...]] = {}
        for row_index, row in featured.iterrows():
            bounds = calculate_price_bounds(
                current_price=float(row["current_price"]),
                base_cost=float(row["base_cost"]),
                competitor_price=float(row["competitor_price"]),
                inventory_level=float(row["inventory_level"]),
                baseline_demand=float(baseline_demand[row_index]),
            )
            candidates = np.linspace(bounds.minimum, bounds.maximum, max(int(n_grid), 5))
            candidates = np.unique(np.append(candidates, float(row["current_price"])))
            scenarios = pd.DataFrame([row.to_dict()] * len(candidates))
            scenarios["current_price"] = np.sort(candidates)
            scenarios["_row_index"] = row_index
            scenario_frames.append(scenarios)
            rule_notes_by_row[row_index] = bounds.notes

        all_scenarios = pd.concat(scenario_frames, ignore_index=True)
        all_scenarios["predicted_demand"] = self.predict_demand(all_scenarios)
        all_scenarios["predicted_revenue"] = all_scenarios["current_price"] * all_scenarios["predicted_demand"]
        all_scenarios["predicted_margin"] = (
            all_scenarios["current_price"] - all_scenarios["base_cost"]
        ) * all_scenarios["predicted_demand"]

        if objective == "revenue":
            all_scenarios["objective_score"] = all_scenarios["predicted_revenue"]
        elif objective == "balanced":
            def normalized(group: pd.DataFrame, column: str) -> pd.Series:
                values = group[column]
                spread = float(values.max() - values.min())
                return pd.Series(np.ones(len(group)), index=group.index) if np.isclose(spread, 0) else (values - values.min()) / spread

            parts = []
            for _, group in all_scenarios.groupby("_row_index", sort=False):
                group = group.copy()
                group["objective_score"] = (
                    0.60 * normalized(group, "predicted_margin")
                    + 0.25 * normalized(group, "predicted_revenue")
                    + 0.15 * normalized(group, "predicted_demand")
                )
                parts.append(group)
            all_scenarios = pd.concat(parts).sort_index()
        else:
            all_scenarios["objective_score"] = all_scenarios["predicted_margin"]

        selected_indices = all_scenarios.groupby("_row_index")["objective_score"].idxmax()
        selected = all_scenarios.loc[selected_indices].sort_values("_row_index").reset_index(drop=True)

        rows: list[dict[str, Any]] = []
        for row_index, row in featured.iterrows():
            best = selected.loc[selected["_row_index"] == row_index].iloc[0]
            current_price = float(row["current_price"] )
            recommended_price = float(best["current_price"] )
            current_demand = float(baseline_demand[row_index])
            optimized_demand = float(best["predicted_demand"] )
            current_revenue = current_price * current_demand
            current_margin = (current_price - float(row["base_cost"])) * current_demand
            optimized_revenue = float(best["predicted_revenue"] )
            optimized_margin = float(best["predicted_margin"] )
            change_pct = (recommended_price - current_price) / max(current_price, 1e-6)
            action = determine_pricing_action(
                current_price=current_price,
                recommended_price=recommended_price,
                inventory_level=float(row["inventory_level"]),
                baseline_demand=current_demand,
                competitor_price=float(row["competitor_price"]),
            )
            recommendation = build_business_recommendation(
                action=action,
                change_pct=change_pct,
                inventory_level=float(row["inventory_level"]),
                optimized_demand=optimized_demand,
                competitor_price=float(row["competitor_price"]),
                recommended_price=recommended_price,
                objective_label=objective_label,
                rule_notes=rule_notes_by_row[row_index],
            )
            segment_id, segment_name = self.assign_segment(row, current_demand)
            rows.append({
                "product_id": row.get("product_id", ""),
                "current_price": current_price,
                "recommended_price": recommended_price,
                "price_change_pct": change_pct,
                "predicted_demand_current": current_demand,
                "predicted_demand_optimized": optimized_demand,
                "predicted_revenue_current": current_revenue,
                "predicted_revenue_optimized": optimized_revenue,
                "predicted_margin_current": current_margin,
                "predicted_margin_optimized": optimized_margin,
                "estimated_margin_uplift": optimized_margin - current_margin,
                "high_demand_probability": float(high_probabilities[row_index]),
                "pricing_segment_id": segment_id,
                "pricing_segment": segment_name,
                "pricing_action": action,
                "business_recommendation": recommendation,
                "objective": objective_label,
            })
        return pd.DataFrame(rows)
