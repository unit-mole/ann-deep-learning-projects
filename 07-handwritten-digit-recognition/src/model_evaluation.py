"""Evaluate a saved ANN and regenerate all portfolio output artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data_preprocessing import prepare_mnist
from src.prediction_pipeline import load_digit_model


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(cm, cmap='Blues')
    ax.set_title('MNIST Test Confusion Matrix')
    ax.set_xlabel('Predicted digit')
    ax.set_ylabel('True digit')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    threshold = cm.max() / 2
    for row in range(10):
        for column in range(10):
            ax.text(
                column,
                row,
                str(cm[row, column]),
                ha='center',
                va='center',
                color='white' if cm[row, column] > threshold else 'black',
                fontsize=8,
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_prediction_grid(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    indexes: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    count = min(10, len(indexes))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for axis in axes.flat:
        axis.axis('off')
    for axis, index in zip(axes.flat, indexes[:count]):
        axis.imshow(images[index], cmap='gray')
        axis.set_title(f'True: {true_labels[index]} | Pred: {predicted_labels[index]}')
        axis.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_training_history(history_path: Path, output_path: Path) -> None:
    if not history_path.exists():
        return
    history = json.loads(history_path.read_text(encoding='utf-8'))
    epochs = np.arange(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history['accuracy'], label='Train')
    axes[0].plot(epochs, history['val_accuracy'], label='Validation')
    axes[0].set(title='Accuracy by Epoch', xlabel='Epoch', ylabel='Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, history['loss'], label='Train')
    axes[1].plot(epochs, history['val_loss'], label='Validation')
    axes[1].set(title='Loss by Epoch', xlabel='Epoch', ylabel='Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def evaluate_model(model_path: Path, output_dir: Path, history_path: Path | None = None) -> dict[str, Any]:
    data = prepare_mnist()
    model = load_digit_model(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_loss, test_accuracy = model.evaluate(data.x_test, data.y_test_one_hot, verbose=0)
    probabilities = model.predict(data.x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    cm = confusion_matrix(data.y_test, predictions)
    report = classification_report(data.y_test, predictions, output_dict=True, digits=4)

    pd.DataFrame(cm, index=range(10), columns=range(10)).to_csv(
        output_dir / 'confusion_matrix.csv'
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / 'classification_report.csv')
    per_class = pd.DataFrame({
        'digit': range(10),
        'correct_predictions': np.diag(cm),
        'support': cm.sum(axis=1),
        'per_class_accuracy': np.diag(cm) / cm.sum(axis=1),
    })
    per_class.to_csv(output_dir / 'per_class_accuracy.csv', index=False)

    plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png')
    correct_indexes = np.where(predictions == data.y_test)[0]
    incorrect_indexes = np.where(predictions != data.y_test)[0]
    plot_prediction_grid(
        data.test_images, data.y_test, predictions, correct_indexes,
        output_dir / 'sample_predictions.png', 'Correct MNIST Predictions'
    )
    plot_prediction_grid(
        data.test_images, data.y_test, predictions, incorrect_indexes,
        output_dir / 'misclassified_digits.png', 'Misclassified MNIST Digits'
    )
    if history_path is not None:
        plot_training_history(history_path, output_dir / 'accuracy_loss_curve.png')

    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'macro_precision': float(report['macro avg']['precision']),
        'macro_recall': float(report['macro avg']['recall']),
        'macro_f1': float(report['macro avg']['f1-score']),
        'weighted_f1': float(report['weighted avg']['f1-score']),
        'correct_predictions': int((predictions == data.y_test).sum()),
        'misclassified_predictions': int((predictions != data.y_test).sum()),
    }
    (output_dir / 'model_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/digit_recognition_model.keras'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--history', type=Path, default=Path('outputs/training_history.json'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_model(args.model, args.output_dir, args.history)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
