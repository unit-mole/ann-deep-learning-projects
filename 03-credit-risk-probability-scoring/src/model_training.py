from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .config import MODEL_DIR, OUTPUT_DIR, SEED
from .data_preprocessing import (
    export_portable_schema, fit_transform_splits, load_dataset,
    save_preprocessor, split_dataset,
)


def set_reproducibility(seed: int = SEED) -> None:
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def build_credit_risk_ann(input_dim: int):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="credit_features"),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.30),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.10),
        layers.Dense(1, activation="sigmoid", name="probability_of_default"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def train_model(dataset_path: str | Path | None = None, epochs: int = 40, batch_size: int = 256):
    from tensorflow.keras import callbacks

    set_reproducibility()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(data)
    preprocessor, X_train_p, X_val_p, X_test_p = fit_transform_splits(X_train, X_val, X_test)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}

    model = build_credit_risk_ann(X_train_p.shape[1])
    callback_list = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(str(MODEL_DIR / "best_credit_risk_ann_model.keras"), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    ]
    history = model.fit(
        X_train_p, np.asarray(y_train).astype("float32"),
        validation_data=(X_val_p, np.asarray(y_val).astype("float32")),
        epochs=epochs, batch_size=batch_size, class_weight=class_weight,
        callbacks=callback_list, verbose=1,
    )
    model.save(MODEL_DIR / "final_credit_risk_ann_model.keras")
    save_preprocessor(preprocessor, MODEL_DIR / "preprocessor.joblib")
    export_portable_schema(preprocessor, MODEL_DIR / "preprocessing_schema.json")
    pd.DataFrame(history.history).to_csv(OUTPUT_DIR / "training_logs.csv", index=False)

    training_metadata = {
        "seed": SEED, "input_dim": int(X_train_p.shape[1]),
        "class_weights": class_weight, "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "train_rows": len(X_train), "validation_rows": len(X_val), "test_rows": len(X_test),
    }
    (MODEL_DIR / "training_metadata.json").write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")
    return model, preprocessor, (X_val, X_val_p, y_val), (X_test, X_test_p, y_test), history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the credit-risk ANN model.")
    parser.add_argument("--data", type=str, default=None, help="Optional CSV containing training data and default_flag target.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    train_model(args.data, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
