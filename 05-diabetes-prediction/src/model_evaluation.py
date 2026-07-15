"""Evaluation metrics, charts, and lightweight permutation importance."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
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

from .config import FEATURE_COLUMNS


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    """Calculate screening-focused metrics at a selected threshold."""
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)
    metrics = {
        "classification_threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }
    return metrics, predictions


def select_screening_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, pd.DataFrame]:
    """Select a validation threshold by maximizing F-beta (beta=2 favors recall)."""
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    rows = []
    for threshold in np.arange(0.15, 0.71, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        denominator = (beta**2 * precision) + recall
        f_beta = (
            (1 + beta**2) * precision * recall / denominator
            if denominator > 0 else 0.0
        )
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": float(precision),
                "recall": float(recall),
                "f_beta": float(f_beta),
                "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
            }
        )
    results = pd.DataFrame(rows)
    best = results.sort_values(
        ["f_beta", "precision", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    return float(best["threshold"]), results


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix)
    ax.set_title("Confusion Matrix — Tuned Screening Threshold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks([0, 1], labels=["No diabetes", "Diabetes"])
    ax.set_yticks([0, 1], labels=["No diabetes", "Diabetes"])
    for row in range(2):
        for column in range(2):
            ax.text(column, row, int(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, probabilities, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ANN (ROC-AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set(title="Receiver Operating Characteristic", xlabel="False-positive rate", ylabel="True-positive rate")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(y_true, probabilities, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"ANN (PR-AUC = {pr_auc:.3f})")
    ax.axhline(prevalence, linestyle="--", label=f"Prevalence = {prevalence:.3f}")
    ax.set(title="Precision–Recall Curve", xlabel="Recall", ylabel="Precision")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_risk_distribution(probabilities: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(probabilities, bins=20, edgecolor="white")
    ax.axvline(0.30, linestyle="--", label="Low/Medium boundary (0.30)")
    ax.axvline(0.60, linestyle="--", label="Medium/High boundary (0.60)")
    ax.set(title="Test-Set Diabetes Risk Probability Distribution", xlabel="Predicted probability", ylabel="Patient count")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def permutation_importance_by_auc(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predict_probability: Callable[[pd.DataFrame], np.ndarray],
    random_state: int = 42,
    repeats: int = 20,
) -> pd.DataFrame:
    """Measure mean ROC-AUC decrease after shuffling each original feature."""
    rng = np.random.default_rng(random_state)
    baseline = roc_auc_score(y_test, predict_probability(X_test))
    rows = []
    for feature in FEATURE_COLUMNS:
        decreases = []
        for _ in range(repeats):
            shuffled = X_test.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            shuffled_auc = roc_auc_score(y_test, predict_probability(shuffled))
            decreases.append(baseline - shuffled_auc)
        rows.append(
            {
                "feature": feature,
                "importance_mean_auc_decrease": float(np.mean(decreases)),
                "importance_std": float(np.std(decreases)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "importance_mean_auc_decrease", ascending=False
    ).reset_index(drop=True)


def save_feature_importance_chart(importance: pd.DataFrame, output_path: Path) -> None:
    ordered = importance.sort_values("importance_mean_auc_decrease")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered["feature"], ordered["importance_mean_auc_decrease"], xerr=ordered["importance_std"])
    ax.set(title="Permutation Importance on Held-Out Test Data", xlabel="Mean decrease in ROC-AUC after shuffling")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
