from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)


def select_threshold(y_true, probabilities, start: float = .05, stop: float = .95, step: float = .01) -> float:
    thresholds = np.arange(start, stop + step, step)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate_outputs(y_churn, churn_probability, y_clv, predicted_clv, y_engagement, predicted_engagement, threshold: float) -> dict:
    churn_label = (churn_probability >= threshold).astype(int)
    return {
        "threshold": threshold,
        "churn": {
            "accuracy": float(accuracy_score(y_churn, churn_label)),
            "precision": float(precision_score(y_churn, churn_label, zero_division=0)),
            "recall": float(recall_score(y_churn, churn_label, zero_division=0)),
            "f1": float(f1_score(y_churn, churn_label, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_churn, churn_probability)),
            "pr_auc": float(average_precision_score(y_churn, churn_probability)),
            "confusion_matrix": confusion_matrix(y_churn, churn_label).tolist(),
        },
        "clv": {
            "mae": float(mean_absolute_error(y_clv, predicted_clv)),
            "rmse": float(np.sqrt(mean_squared_error(y_clv, predicted_clv))),
            "r2": float(r2_score(y_clv, predicted_clv)),
        },
        "engagement": {
            "mae": float(mean_absolute_error(y_engagement, predicted_engagement)),
            "rmse": float(np.sqrt(mean_squared_error(y_engagement, predicted_engagement))),
            "r2": float(r2_score(y_engagement, predicted_engagement)),
        },
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
