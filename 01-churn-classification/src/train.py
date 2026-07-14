from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    ASSET_DIR,
    DATA_DIR,
    IDENTIFIER_COLUMNS,
    MODEL_DIR,
    REQUIRED_FEATURES,
    TARGET_COLUMN,
)
from src.evaluation import (
    calculate_metrics,
    choose_f1_threshold,
    save_evaluation_plots,
)
from src.modeling import build_ann
from src.preprocessing import build_preprocessor

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)


def load_training_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path)

    expected = set(REQUIRED_FEATURES + [TARGET_COLUMN])
    missing = sorted(expected - set(data.columns))
    if missing:
        raise ValueError("Training data is missing: " + ", ".join(missing))

    data = data.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")
    features = data.loc[:, REQUIRED_FEATURES].copy()
    target = data[TARGET_COLUMN].astype(int)

    return features, target


def train(
    csv_path: Path,
    model_dir: Path,
    asset_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    set_seed(seed)
    features, target = load_training_data(csv_path)

    # 70% train, 15% validation, 15% test.
    train_validation_x, test_x, train_validation_y, test_y = train_test_split(
        features,
        target,
        test_size=0.15,
        random_state=seed,
        stratify=target,
    )
    validation_fraction_of_remaining = 0.15 / 0.85
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_validation_x,
        train_validation_y,
        test_size=validation_fraction_of_remaining,
        random_state=seed,
        stratify=train_validation_y,
    )

    preprocessor = build_preprocessor()
    train_processed = preprocessor.fit_transform(train_x)
    validation_processed = preprocessor.transform(validation_x)
    test_processed = preprocessor.transform(test_x)

    class_labels = np.unique(train_y)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=train_y,
    )
    class_weights = {
        int(label): float(weight)
        for label, weight in zip(class_labels, class_weight_values)
    }

    model = build_ann(input_dimension=train_processed.shape[1])

    callbacks = [
        EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=15,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        train_processed,
        train_y,
        validation_data=(validation_processed, validation_y),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    validation_probabilities = model.predict(
        validation_processed, verbose=0
    ).reshape(-1)
    threshold = choose_f1_threshold(
        validation_y,
        validation_probabilities,
    )

    test_probabilities = model.predict(test_processed, verbose=0).reshape(-1)
    test_metrics = calculate_metrics(test_y, test_probabilities, threshold)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model.keras")
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(model_dir / "training_history.csv", index=False)

    metadata = {
        "project_name": "ANN Churn Classification",
        "artifact_mode": "modern_pipeline",
        "model_format": "Keras v3",
        "default_threshold": threshold,
        "dataset": {
            "rows": int(len(features)),
            "target": TARGET_COLUMN,
            "churn_rate": float(target.mean()),
        },
        "split": {
            "train_rows": int(len(train_x)),
            "validation_rows": int(len(validation_x)),
            "test_rows": int(len(test_x)),
            "stratified": True,
            "random_seed": seed,
        },
        "model": {
            "architecture": [
                "Dense(64, ReLU)",
                "Dropout(0.20)",
                "Dense(16, ReLU)",
                "Dense(1, Sigmoid)",
            ],
            "class_weights": class_weights,
            "epochs_requested": epochs,
            "epochs_completed": int(len(history_frame)),
            "batch_size": batch_size,
        },
        "evaluation": test_metrics,
        "threshold_selection": (
            "Threshold selected on the validation set by maximum churn-class F1."
        ),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    save_evaluation_plots(
        test_y,
        test_probabilities,
        threshold,
        asset_dir,
    )
    return metadata


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the portfolio-ready ANN churn classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "raw" / "Churn_Modelling.csv",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR / "charts")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    result = train(
        csv_path=arguments.data,
        model_dir=arguments.model_dir,
        asset_dir=arguments.asset_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2))
