from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)


def tune_threshold(y_true, probabilities, objective: str = "f1") -> tuple[float, pd.DataFrame]:
    records = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        labels = (probabilities >= threshold).astype(int)
        records.append({
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(y_true, labels),
            "precision": precision_score(y_true, labels, zero_division=0),
            "recall": recall_score(y_true, labels, zero_division=0),
            "f1": f1_score(y_true, labels, zero_division=0),
        })
    table = pd.DataFrame(records)
    if objective not in table.columns:
        raise ValueError(f"Unsupported threshold objective: {objective}")
    best = float(table.sort_values([objective, "precision", "recall"], ascending=False).iloc[0]["threshold"])
    return best, table


def compute_metrics(y_true, probabilities, threshold: float) -> dict:
    labels = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "classification_threshold": threshold,
        "accuracy": accuracy_score(y_true, labels),
        "precision": precision_score(y_true, labels, zero_division=0),
        "recall": recall_score(y_true, labels, zero_division=0),
        "f1": f1_score(y_true, labels, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, labels).tolist(),
        "classification_report": classification_report(y_true, labels, output_dict=True, zero_division=0),
    }


def save_evaluation_plots(y_true, probabilities, threshold: float, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = (np.asarray(probabilities) >= threshold).astype(int)

    cm = confusion_matrix(y_true, labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)
    ax.set(title="Confusion Matrix", xlabel="Predicted Label", ylabel="True Label", xticks=[0, 1], yticks=[0, 1])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "roc_curve.png", dpi=160); plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, probabilities):.3f}")
    ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "precision_recall_curve.png", dpi=160); plt.close(fig)

    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(predicted, observed, marker="o", label="ANN")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set(title="Calibration Curve", xlabel="Predicted Probability", ylabel="Observed Default Rate")
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "calibration_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(probabilities, bins=25)
    ax.set(title="Predicted Risk Score Distribution", xlabel="Predicted Probability of Default", ylabel="Applicants")
    fig.tight_layout(); fig.savefig(output_dir / "risk_score_distribution.png", dpi=160); plt.close(fig)


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
