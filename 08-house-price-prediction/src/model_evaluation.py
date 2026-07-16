"""Regression evaluation and portfolio-ready visual outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

try:
    from .config import FEATURE_COLUMNS, TARGET_MULTIPLIER
except ImportError:
    from config import FEATURE_COLUMNS, TARGET_MULTIPLIER


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics in model units and US dollars."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae_target_units": mae,
        "mae_usd": mae * TARGET_MULTIPLIER,
        "rmse_target_units": rmse,
        "rmse_usd": rmse * TARGET_MULTIPLIER,
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "mape_percent": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def empirical_error_band(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coverage: float = 0.80,
) -> float:
    """Return a symmetric absolute-error half-width in model target units."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1.")
    absolute_errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float(np.quantile(absolute_errors, coverage))


def permutation_importance_ann(
    model: Any,
    scaler: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure each feature's importance as the increase in test RMSE."""
    rng = np.random.default_rng(random_state)
    ordered = x_test[FEATURE_COLUMNS].reset_index(drop=True)
    y_array = np.asarray(y_test).ravel()

    baseline_pred = model.predict(scaler.transform(ordered), verbose=0).ravel()
    baseline_rmse = float(np.sqrt(mean_squared_error(y_array, baseline_pred)))

    rows: list[dict[str, float | str]] = []
    for column in FEATURE_COLUMNS:
        increases: list[float] = []
        for _ in range(repeats):
            shuffled = ordered.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            prediction = model.predict(
                scaler.transform(shuffled), verbose=0
            ).ravel()
            shuffled_rmse = float(np.sqrt(mean_squared_error(y_array, prediction)))
            increases.append(shuffled_rmse - baseline_rmse)

        rows.append(
            {
                "feature": column,
                "rmse_increase_target_units": float(np.mean(increases)),
                "rmse_increase_usd": float(np.mean(increases) * TARGET_MULTIPLIER),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "rmse_increase_target_units", ascending=False
    )


def save_evaluation_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    importance: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Save metrics and standard regression charts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_true - y_pred
    metrics = regression_metrics(y_true, y_pred)

    with (output_path / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    plt.figure(figsize=(7, 5))
    plt.scatter(
        y_true * TARGET_MULTIPLIER,
        y_pred * TARGET_MULTIPLIER,
        alpha=0.35,
    )
    limits = [
        min(y_true.min(), y_pred.min()) * TARGET_MULTIPLIER,
        max(y_true.max(), y_pred.max()) * TARGET_MULTIPLIER,
    ]
    plt.plot(limits, limits, linestyle="--")
    plt.xlabel("Actual median house value (USD)")
    plt.ylabel("Predicted median house value (USD)")
    plt.title("Actual vs Predicted Values")
    plt.tight_layout()
    plt.savefig(output_path / "actual_vs_predicted.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred * TARGET_MULTIPLIER, residuals * TARGET_MULTIPLIER, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted median house value (USD)")
    plt.ylabel("Residual: actual - predicted (USD)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(output_path / "residual_plot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(residuals * TARGET_MULTIPLIER, bins=40)
    plt.xlabel("Prediction error (USD)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.savefig(output_path / "error_distribution.png", dpi=180)
    plt.close()

    if importance is not None and not importance.empty:
        chart = importance.sort_values("rmse_increase_usd")
        plt.figure(figsize=(7, 4.8))
        plt.barh(chart["feature"], chart["rmse_increase_usd"])
        plt.xlabel("Increase in test RMSE after permutation (USD)")
        plt.title("ANN Permutation Feature Importance")
        plt.tight_layout()
        plt.savefig(output_path / "feature_importance.png", dpi=180)
        plt.close()
        importance.to_csv(output_path / "feature_importance.csv", index=False)

    return metrics
