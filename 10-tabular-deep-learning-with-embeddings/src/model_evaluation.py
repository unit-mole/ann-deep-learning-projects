from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary_classifier(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    """Calculate threshold metrics and probability-ranking metrics."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "log_loss": float(log_loss(target, probabilities)),
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
        "classification_report": classification_report(
            target, predictions, output_dict=True, zero_division=0
        ),
    }


def select_f1_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    minimum: float = 0.10,
    maximum: float = 0.90,
    step: float = 0.01,
) -> tuple[float, pd.DataFrame]:
    """Select the highest-F1 threshold using validation data only."""
    rows = []
    for threshold in np.arange(minimum, maximum + step / 2, step):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "accuracy": accuracy_score(target, predictions),
                "precision": precision_score(target, predictions, zero_division=0),
                "recall": recall_score(target, predictions, zero_division=0),
                "f1": f1_score(target, predictions, zero_division=0),
            }
        )
    analysis = pd.DataFrame(rows)
    best_threshold = float(analysis.loc[analysis["f1"].idxmax(), "threshold"])
    return best_threshold, analysis


def save_evaluation_plots(
    target: np.ndarray,
    probabilities: np.ndarray,
    output_directory: Path,
    threshold: float = 0.50,
) -> None:
    """Save confusion matrix, ROC curve, and precision–recall curve."""
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(target, predictions)

    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Lower", "Higher"])
    ax.set_yticks([0, 1], ["Lower", "Higher"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_directory / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(target, probabilities)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"AUC = {roc_auc_score(target, probabilities):.3f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "roc_curve.png", dpi=180)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(target, probabilities)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        recall,
        precision,
        label=f"AP = {average_precision_score(target, probabilities):.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "precision_recall_curve.png", dpi=180)
    plt.close(fig)


def save_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    """Save loss and AUC curves without requiring notebook execution."""
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(frame.index + 1, frame["loss"], label="Train")
    if "val_loss" in frame:
        axes[0].plot(frame.index + 1, frame["val_loss"], label="Validation")
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    metric_name = "auc" if "auc" in frame else "accuracy"
    validation_metric = f"val_{metric_name}"
    axes[1].plot(frame.index + 1, frame[metric_name], label="Train")
    if validation_metric in frame:
        axes[1].plot(frame.index + 1, frame[validation_metric], label="Validation")
    axes[1].set_title(metric_name.upper().replace("_", " "))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_name.upper())
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_model_comparison(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    """Save a recruiter-friendly comparison of decision and ranking metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    available = [metric for metric in metrics if metric in metrics_frame.columns]
    chart = metrics_frame.set_index("model")[available].T
    ax = chart.plot(kind="bar", figsize=(10, 5.5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_title("Embedding ANN vs One-Hot Random Forest")
    ax.legend(title="Model")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def save_permutation_importance_plot(
    importance_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot raw-feature ROC-AUC decrease after validation-set shuffling."""
    ordered = importance_frame.sort_values("roc_auc_decrease")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.barh(ordered["feature"], ordered["roc_auc_decrease"])
    ax.set_xlabel("ROC-AUC decrease after shuffling")
    ax.set_ylabel("Feature")
    ax.set_title("Raw-Feature Permutation Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_prediction_distribution(
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Save a simple probability distribution for the held-out test set."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(probabilities, bins=25)
    ax.set_xlabel("Predicted positive probability")
    ax.set_ylabel("Rows")
    ax.set_title("Prediction Probability Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_metrics(metrics: dict, path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
