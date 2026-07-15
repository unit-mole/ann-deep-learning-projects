from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    nonzero = y_true != 0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_nonzero_actuals_pct": float(np.mean(np.abs(residual[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else float("nan"),
        "smape_pct": float(np.mean(2 * np.abs(residual) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100),
        "wape_pct": float(np.sum(np.abs(residual)) / max(np.sum(np.abs(y_true)), 1e-8) * 100),
    }


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.50) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    labels = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def numeric_permutation_importance(
    model,
    base_inputs: dict[str, np.ndarray],
    y_true: np.ndarray,
    numeric_feature_names: list[str],
    reverse_target_transform,
    repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure MAE degradation after shuffling each numeric feature."""
    rng = np.random.default_rng(seed)
    baseline_raw = model.predict(base_inputs, verbose=0)[0].reshape(-1)
    baseline = reverse_target_transform(baseline_raw)
    baseline_mae = mean_absolute_error(y_true, baseline)
    rows = []
    for index, feature in enumerate(numeric_feature_names):
        changes = []
        for _ in range(repeats):
            perturbed = {key: value.copy() for key, value in base_inputs.items()}
            shuffled = perturbed["numeric_input"][:, index].copy()
            rng.shuffle(shuffled)
            perturbed["numeric_input"][:, index] = shuffled
            prediction_raw = model.predict(perturbed, verbose=0)[0].reshape(-1)
            prediction = reverse_target_transform(prediction_raw)
            changes.append(mean_absolute_error(y_true, prediction) - baseline_mae)
        rows.append({
            "feature": feature,
            "mae_increase_mean": float(np.mean(changes)),
            "mae_increase_std": float(np.std(changes)),
        })
    return pd.DataFrame(rows).sort_values("mae_increase_mean", ascending=False).reset_index(drop=True)
