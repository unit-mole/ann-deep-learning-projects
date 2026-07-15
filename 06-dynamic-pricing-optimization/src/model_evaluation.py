"""Generate evaluation assets for the README and portfolio."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .constants import DEFAULT_RECORD
from .prediction_pipeline import DynamicPricingPipeline


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def create_evaluation_outputs(model_dir: str | Path, output_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = joblib.load(model_dir / "evaluation_payload.joblib")
    metrics = payload["metrics"]
    y_true = np.asarray(payload["true_demand"])
    y_pred = np.asarray(payload["predicted_demand"])
    test_frame = payload["test_frame"].copy()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.28, s=18)
    line_min = min(y_true.min(), y_pred.min())
    line_max = max(y_true.max(), y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--")
    plt.title("Actual vs Predicted Demand")
    plt.xlabel("Actual demand (units)")
    plt.ylabel("Predicted demand (units)")
    _save_figure(output_dir / "actual_vs_predicted_demand.png")

    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.28, s=18)
    plt.axhline(0, linestyle="--")
    plt.title("Demand Prediction Residuals")
    plt.xlabel("Predicted demand (units)")
    plt.ylabel("Actual - predicted")
    _save_figure(output_dir / "residual_plot.png")

    history = payload["history_regression"]
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.title("ANN Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss on log-demand")
    plt.legend()
    _save_figure(output_dir / "training_history.png")

    pipeline = DynamicPricingPipeline(model_dir)
    demo = pipeline.recommend_one(DEFAULT_RECORD, objective_label="Maximize Margin", n_grid=41)
    scenarios = demo["scenario_table"].copy()
    scenarios.to_csv(output_dir / "price_scenario_example.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(scenarios["current_price"], scenarios["predicted_demand"], marker="o", markersize=3)
    plt.axvline(demo["recommended_price"], linestyle="--", label="Recommended price")
    plt.title("Price vs Predicted Demand")
    plt.xlabel("Candidate price")
    plt.ylabel("Predicted demand (units)")
    plt.legend()
    _save_figure(output_dir / "price_vs_demand_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(scenarios["current_price"], scenarios["predicted_revenue"], label="Revenue")
    plt.plot(scenarios["current_price"], scenarios["predicted_margin"], label="Gross margin")
    plt.axvline(demo["recommended_price"], linestyle="--", label="Recommended price")
    plt.title("Revenue and Margin Optimization Curves")
    plt.xlabel("Candidate price")
    plt.ylabel("Estimated value")
    plt.legend()
    _save_figure(output_dir / "revenue_optimization_curve.png")

    sample = test_frame.head(50).copy()
    optimized = pipeline.recommend_batch(sample, objective_label="Maximize Margin", n_grid=13)
    optimized.to_csv(output_dir / "example_optimized_output.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.hist(optimized["recommended_price"], bins=28)
    plt.title("Distribution of Recommended Prices")
    plt.xlabel("Recommended price")
    plt.ylabel("Products")
    _save_figure(output_dir / "optimized_price_distribution.png")

    baseline_mae = mean_absolute_error(y_true, y_pred)
    rng = np.random.default_rng(42)
    sample_for_importance = test_frame.sample(min(150, len(test_frame)), random_state=42).reset_index(drop=True)
    base_predictions = pipeline.predict_demand(sample_for_importance)
    actual = sample_for_importance["realized_demand"].to_numpy()
    base_error = mean_absolute_error(actual, base_predictions)
    importances: list[dict[str, float | str]] = []
    for column in ["day_of_year", "promotion_flag", "base_cost", "current_price", "competitor_price", "rating", "inventory_level", "marketing_index", "demand_index", "historical_sales"]:
        permuted = sample_for_importance.copy()
        # Shuffle within product category to avoid unrealistic cross-category price/cost combinations.
        permuted[column] = permuted.groupby("category", group_keys=False)[column].transform(
            lambda values: pd.Series(rng.permutation(values.to_numpy()), index=values.index)
        )
        error = mean_absolute_error(actual, pipeline.predict_demand(permuted))
        importances.append({"feature": column, "mae_increase": max(float(error - base_error), 0.0)})
    importance_df = pd.DataFrame(importances).sort_values("mae_increase", ascending=False)
    importance_df.to_csv(output_dir / "permutation_importance.csv", index=False)

    top = importance_df.head(12).sort_values("mae_increase")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mae_increase"])
    plt.title("Permutation Importance: Top Demand Drivers")
    plt.xlabel("Increase in MAE after permutation")
    _save_figure(output_dir / "feature_importance.png")

    metrics["portfolio_outputs"] = {
        "baseline_mae_recalculated": float(baseline_mae),
        "demo_recommended_price": float(demo["recommended_price"]),
        "demo_pricing_action": demo["pricing_action"],
        "demo_estimated_margin_uplift": float(demo["estimated_margin_uplift"]),
    }
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
