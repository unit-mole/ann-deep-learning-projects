"""Evaluation utilities for imbalanced binary fraud classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def select_decision_threshold(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    *,
    strategy: Literal["max_f1", "recall_target"] = "max_f1",
    recall_target: float = 0.85,
) -> dict[str, float | str]:
    """Select a decision threshold using validation data only.

    ``max_f1`` balances precision and recall.
    ``recall_target`` finds the highest-precision threshold that meets the
    specified validation recall target.
    """
    y_array = np.asarray(y_true).astype(int)
    probability_array = np.asarray(probabilities, dtype=float).ravel()
    precision, recall, thresholds = precision_recall_curve(
        y_array, probability_array
    )

    if thresholds.size == 0:
        raise ValueError("Threshold selection requires both target classes.")

    precision_at_threshold = precision[:-1]
    recall_at_threshold = recall[:-1]
    f1_values = (
        2
        * precision_at_threshold
        * recall_at_threshold
        / (precision_at_threshold + recall_at_threshold + 1e-12)
    )

    if strategy == "max_f1":
        selected_index = int(np.nanargmax(f1_values))
    elif strategy == "recall_target":
        eligible = np.flatnonzero(recall_at_threshold >= recall_target)
        if eligible.size == 0:
            selected_index = int(np.nanargmax(recall_at_threshold))
        else:
            selected_index = int(
                eligible[np.nanargmax(precision_at_threshold[eligible])]
            )
    else:
        raise ValueError(f"Unsupported threshold strategy: {strategy}")

    return {
        "strategy": strategy,
        "threshold": float(thresholds[selected_index]),
        "validation_precision": float(precision_at_threshold[selected_index]),
        "validation_recall": float(recall_at_threshold[selected_index]),
        "validation_f1": float(f1_values[selected_index]),
        "recall_target": float(recall_target),
    }


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Calculate threshold-dependent and ranking metrics."""
    y_array = np.asarray(y_true).astype(int)
    probability_array = np.asarray(probabilities, dtype=float).ravel()

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if y_array.shape[0] != probability_array.shape[0]:
        raise ValueError("y_true and probabilities must have equal length.")

    predictions = (probability_array >= threshold).astype(int)
    matrix = confusion_matrix(y_array, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_array, predictions)),
        "precision": float(
            precision_score(y_array, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_array, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_array, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_array, probability_array)),
        "pr_auc": float(
            average_precision_score(y_array, probability_array)
        ),
        "confusion_matrix": matrix.tolist(),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
        "actual_fraud": int(y_array.sum()),
        "predicted_fraud": int(predictions.sum()),
        "evaluated_rows": int(y_array.shape[0]),
        "classification_report": classification_report(
            y_array,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            output_dict=True,
            zero_division=0,
        ),
    }


def save_evaluation_plots(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
    output_dir: str | Path,
) -> list[Path]:
    """Save confusion-matrix, ROC, and precision-recall figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    y_array = np.asarray(y_true).astype(int)
    probability_array = np.asarray(probabilities, dtype=float).ravel()
    predictions = (probability_array >= threshold).astype(int)

    saved_paths: list[Path] = []

    matrix = confusion_matrix(y_array, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix)
    axis.set_xticks([0, 1], labels=["Predicted Safe", "Predicted Fraud"])
    axis.set_yticks([0, 1], labels=["Actual Safe", "Actual Fraud"])
    axis.set_title(f"Confusion Matrix — Threshold {threshold:.3f}")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    for row_index in range(2):
        for column_index in range(2):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:,}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    confusion_path = output_path / "confusion_matrix.png"
    figure.savefig(confusion_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    saved_paths.append(confusion_path)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_array, probability_array
    )
    roc_auc = roc_auc_score(y_array, probability_array)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ANN (ROC-AUC = {roc_auc:.4f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    axis.set_title("ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    roc_path = output_path / "roc_curve.png"
    figure.savefig(roc_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    saved_paths.append(roc_path)

    precision, recall, _ = precision_recall_curve(y_array, probability_array)
    pr_auc = average_precision_score(y_array, probability_array)
    baseline = float(y_array.mean())
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label=f"ANN (PR-AUC = {pr_auc:.4f})")
    axis.axhline(
        baseline,
        linestyle="--",
        label=f"Fraud prevalence = {baseline:.4%}",
    )
    axis.set_title("Precision–Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    precision_recall_path = output_path / "precision_recall_curve.png"
    figure.savefig(precision_recall_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    saved_paths.append(precision_recall_path)

    return saved_paths


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> Path:
    """Save metrics as readable JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
