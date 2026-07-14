from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def choose_f1_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
) -> float:
    """Choose a validation-set threshold that maximizes churn-class F1."""
    y_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities)
    thresholds = np.linspace(0.10, 0.90, 161)

    scores = [
        f1_score(y_array, probability_array >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def calculate_metrics(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    y_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities)
    predictions = (probability_array >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_array, predictions)),
        "precision_churn": float(
            precision_score(y_array, predictions, zero_division=0)
        ),
        "recall_churn": float(
            recall_score(y_array, predictions, zero_division=0)
        ),
        "f1_churn": float(f1_score(y_array, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_array, probability_array)),
        "pr_auc": float(average_precision_score(y_array, probability_array)),
        "confusion_matrix": confusion_matrix(y_array, predictions).tolist(),
    }


def save_evaluation_plots(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    y_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities)
    predictions = (probability_array >= threshold).astype(int)

    display = ConfusionMatrixDisplay.from_predictions(
        y_array,
        predictions,
        display_labels=["Retained", "Churned"],
        values_format="d",
    )
    display.ax_.set_title("Test-Set Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(
        output_directory / "modern-confusion-matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(display.figure_)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_array, probability_array
    )
    plt.figure(figsize=(7, 5))
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test-Set ROC Curve")
    plt.tight_layout()
    plt.savefig(
        output_directory / "modern-roc-curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    precision, recall, _ = precision_recall_curve(
        y_array, probability_array
    )
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test-Set Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(
        output_directory / "modern-precision-recall-curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()
