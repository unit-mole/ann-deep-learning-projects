"""Train, evaluate, and save the portfolio-ready Keras ANN."""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .config import DATA_DIR, FEATURE_COLUMNS, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE
from .data_preprocessing import (
    build_preprocessor,
    create_data_quality_report,
    load_dataset,
    split_dataset,
)
from .model_evaluation import (
    classification_metrics,
    permutation_importance_by_auc,
    save_confusion_matrix,
    save_feature_importance_chart,
    save_precision_recall_curve,
    save_risk_distribution,
    save_roc_curve,
    select_screening_threshold,
)
from .risk_scoring import categorize_probabilities


def set_reproducible_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_ann(input_dimension: int) -> tf.keras.Model:
    """Create the tuned 64→32 binary classification network."""
    inputs = tf.keras.Input(shape=(input_dimension,), name="patient_features")
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_64",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout_30")(x)
    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_32",
    )(x)
    x = tf.keras.layers.Dropout(0.15, name="dropout_15")(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="diabetes_probability"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="diabetes_ann")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def train(data_path: Path = DATA_DIR / "diabetes.csv") -> dict:
    set_reproducible_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset(data_path)
    quality_report = create_data_quality_report(data)
    (X_train, X_validation, X_test,
     y_train, y_validation, y_test) = split_dataset(data)

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)
    X_test_transformed = preprocessor.transform(X_test)

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weights = {int(k): float(v) for k, v in zip(classes, weights)}

    model = build_ann(X_train_transformed.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=10,
            min_delta=1e-4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        X_train_transformed,
        y_train,
        validation_data=(X_validation_transformed, y_validation),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )

    validation_probabilities = model.predict(
        X_validation_transformed, verbose=0
    ).ravel()
    threshold, threshold_results = select_screening_threshold(
        y_validation, validation_probabilities, beta=2.0
    )
    threshold_results.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)

    test_probabilities = model.predict(X_test_transformed, verbose=0).ravel()
    test_metrics, test_predictions = classification_metrics(
        y_test, test_probabilities, threshold
    )

    save_confusion_matrix(
        np.asarray(test_metrics["confusion_matrix"]),
        OUTPUT_DIR / "confusion_matrix.png",
    )
    save_roc_curve(y_test, test_probabilities, OUTPUT_DIR / "roc_curve.png")
    save_precision_recall_curve(
        y_test, test_probabilities, OUTPUT_DIR / "precision_recall_curve.png"
    )
    save_risk_distribution(
        test_probabilities, OUTPUT_DIR / "risk_distribution.png"
    )

    def predict_probability(frame: pd.DataFrame) -> np.ndarray:
        transformed = preprocessor.transform(frame)
        return model.predict(transformed, verbose=0).ravel()

    importance = permutation_importance_by_auc(
        X_test, y_test, predict_probability, repeats=20
    )
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    save_feature_importance_chart(
        importance, OUTPUT_DIR / "feature_importance.png"
    )

    scored_test = X_test.reset_index(drop=True).copy()
    scored_test["ActualOutcome"] = y_test.reset_index(drop=True)
    scored_test["DiabetesRiskProbability"] = test_probabilities
    scored_test["PredictedOutcome"] = test_predictions
    scored_test["RiskCategory"] = categorize_probabilities(
        test_probabilities
    ).to_numpy()
    scored_test.to_csv(OUTPUT_DIR / "scored_test_sample.csv", index=False)

    model.save(MODEL_DIR / "diabetes_ann.keras")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    metadata = {
        "model_name": "Diabetes Risk Screening ANN",
        "model_format": "Keras v3 .keras",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "features": FEATURE_COLUMNS,
        "preprocessed_input_dimension": int(X_train_transformed.shape[1]),
        "architecture": [64, 32, 1],
        "activation_functions": ["relu", "relu", "sigmoid"],
        "dropout_rates": [0.30, 0.15],
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs_ran": len(history.history["loss"]),
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "threshold_selection": "Validation-set F2 maximization (recall weighted more than precision)",
        "classification_threshold": threshold,
        "risk_bands": {
            "Low Risk": "probability < 0.30",
            "Medium Risk": "0.30 <= probability < 0.60",
            "High Risk": "probability >= 0.60",
        },
        "data_quality": quality_report,
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_validation)),
            "test": int(len(X_test)),
        },
        "test_metrics": test_metrics,
        "top_permutation_features": importance.head(5).to_dict(orient="records"),
        "limitations": [
            "Small public benchmark dataset with limited demographic scope.",
            "Retrospective portfolio demonstration, not prospective clinical validation.",
            "Risk bands are communication bands and are not clinical decision thresholds.",
        ],
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "model_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "data_quality_report.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history.history).to_csv(
        OUTPUT_DIR / "training_history.csv", index=False
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "diabetes.csv",
        help="Path to the diabetes CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = train(arguments.data)
    print(json.dumps(result["test_metrics"], indent=2))
