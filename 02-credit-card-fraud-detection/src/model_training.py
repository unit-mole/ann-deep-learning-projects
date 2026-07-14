"""Train, validate, evaluate, and save an ANN fraud-detection model."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    FEATURE_COLUMNS,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
)
from src.data_preprocessing import (
    fit_and_transform,
    load_dataset,
    save_feature_schema,
    split_dataset,
)
from src.model_evaluation import (
    calculate_metrics,
    save_evaluation_plots,
    save_metrics,
    select_decision_threshold,
)


DEFAULT_PARAMS: dict[str, float | int] = {
    "hidden_units": 64,
    "dropout_rate": 0.20,
    "learning_rate": 0.001,
    "batch_size": 256,
}


def require_tensorflow():
    """Import TensorFlow with a clear installation error."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training. Install project dependencies "
            "with 'pip install -r requirements.txt'."
        ) from exc
    return tf


def set_reproducible_seeds(seed: int = RANDOM_STATE) -> None:
    """Set deterministic seeds where supported."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf = require_tensorflow()
    tf.random.set_seed(seed)


def build_ann(
    *,
    input_dim: int,
    hidden_units: int = 64,
    dropout_rate: float = 0.20,
    learning_rate: float = 0.001,
):
    """Build a compact feed-forward ANN for tabular binary classification."""
    tf = require_tensorflow()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="transaction_features"),
            tf.keras.layers.Dense(
                hidden_units,
                activation="relu",
                kernel_initializer="he_normal",
                name="hidden_1",
            ),
            tf.keras.layers.BatchNormalization(name="batch_norm_1"),
            tf.keras.layers.Dropout(dropout_rate, name="dropout_1"),
            tf.keras.layers.Dense(
                max(hidden_units // 2, 8),
                activation="relu",
                kernel_initializer="he_normal",
                name="hidden_2",
            ),
            tf.keras.layers.BatchNormalization(name="batch_norm_2"),
            tf.keras.layers.Dropout(dropout_rate, name="dropout_2"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="fraud_probability"),
        ],
        name="credit_card_fraud_ann",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Calculate inverse-frequency class weights from training labels only."""
    classes = np.array([0, 1], dtype=int)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.asarray(labels).astype(int),
    )
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def load_hyperparameters(path: str | Path | None) -> dict[str, Any]:
    """Load supplied hyperparameters or use the documented defaults."""
    if path is None:
        return DEFAULT_PARAMS.copy()

    parameter_path = Path(path)
    if not parameter_path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {parameter_path}")

    params = DEFAULT_PARAMS.copy()
    params.update(json.loads(parameter_path.read_text(encoding="utf-8")))
    return params


def train(
    *,
    dataset_path: str | Path,
    params_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    output_dir: str | Path = OUTPUT_DIR,
    epochs: int = 40,
    threshold_strategy: str = "max_f1",
    recall_target: float = 0.85,
) -> dict[str, Any]:
    """Execute the end-to-end training workflow."""
    set_reproducible_seeds()
    tf = require_tensorflow()

    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    output_path = Path(output_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset(dataset_path)
    splits = split_dataset(dataframe)
    scaler, X_train, X_valid, X_test = fit_and_transform(splits)
    parameters = load_hyperparameters(params_path)

    # Build a fresh model for final training. This avoids continuing training
    # from a hyperparameter-search candidate that already saw validation data.
    model = build_ann(
        input_dim=len(FEATURE_COLUMNS),
        hidden_units=int(parameters["hidden_units"]),
        dropout_rate=float(parameters["dropout_rate"]),
        learning_rate=float(parameters["learning_rate"]),
    )

    class_weights = balanced_class_weights(splits.y_train.to_numpy())
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        splits.y_train.to_numpy(),
        validation_data=(X_valid, splits.y_valid.to_numpy()),
        epochs=epochs,
        batch_size=int(parameters["batch_size"]),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Reload the best checkpoint and use validation data for threshold choice.
    model = tf.keras.models.load_model(model_path, compile=False)
    valid_probabilities = model.predict(
        X_valid, batch_size=4096, verbose=0
    ).ravel()
    test_probabilities = model.predict(
        X_test, batch_size=4096, verbose=0
    ).ravel()

    threshold_selection = select_decision_threshold(
        splits.y_valid,
        valid_probabilities,
        strategy=threshold_strategy,
        recall_target=recall_target,
    )
    threshold = float(threshold_selection["threshold"])
    test_metrics = calculate_metrics(
        splits.y_test,
        test_probabilities,
        threshold=threshold,
    )

    with scaler_path.open("wb") as file:
        pickle.dump(scaler, file)

    save_feature_schema(model_path.parent / "feature_schema.json")
    (model_path.parent / "credit_card_best_params.json").write_text(
        json.dumps(parameters, indent=2),
        encoding="utf-8",
    )
    (model_path.parent / "decision_threshold.json").write_text(
        json.dumps(
            {
                "default_threshold": threshold,
                "selection": threshold_selection,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    complete_metrics = {
        "dataset": {
            "rows": int(dataframe.shape[0]),
            "features": len(FEATURE_COLUMNS),
            "fraud_rate": float(dataframe["Class"].mean()),
        },
        "split": {
            "train_rows": int(len(splits.X_train)),
            "validation_rows": int(len(splits.X_valid)),
            "test_rows": int(len(splits.X_test)),
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "training": {
            "hyperparameters": parameters,
            "class_weights": class_weights,
            "epochs_completed": len(history.history["loss"]),
            "best_validation_pr_auc": float(
                max(history.history.get("val_pr_auc", [float("nan")]))
            ),
        },
        "threshold_selection": threshold_selection,
        "test_metrics": test_metrics,
    }

    save_metrics(complete_metrics, output_path / "model_metrics.json")
    save_evaluation_plots(
        splits.y_test,
        test_probabilities,
        threshold=threshold,
        output_dir=output_path,
    )

    history_path = output_path / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return complete_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/creditcard.csv"),
        help="Path to the full training dataset.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=MODEL_DIR / "credit_card_best_params.json",
        help="JSON file containing ANN hyperparameters.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--threshold-strategy",
        choices=["max_f1", "recall_target"],
        default="max_f1",
    )
    parser.add_argument("--recall-target", type=float, default=0.85)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = train(
        dataset_path=arguments.data,
        params_path=arguments.params,
        epochs=arguments.epochs,
        threshold_strategy=arguments.threshold_strategy,
        recall_target=arguments.recall_target,
    )
    print(json.dumps(result["test_metrics"], indent=2))
