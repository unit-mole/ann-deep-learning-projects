"""Train, tune, evaluate, and save the ANN regression model."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from .config import (
        DATA_PATH,
        FEATURE_COLUMNS,
        METADATA_PATH,
        MODEL_PATH,
        OUTPUT_DIR,
        PARAMS_PATH,
        RANDOM_STATE,
        SCALER_PATH,
        TARGET_MULTIPLIER,
    )
    from .data_preprocessing import (
        build_iqr_outlier_report,
        fit_scaler,
        impute_numeric_values,
        load_housing_data,
        save_scaler,
        split_housing_data,
    )
    from .model_evaluation import (
        empirical_error_band,
        permutation_importance_ann,
        save_evaluation_outputs,
    )
except ImportError:
    from config import (
        DATA_PATH,
        FEATURE_COLUMNS,
        METADATA_PATH,
        MODEL_PATH,
        OUTPUT_DIR,
        PARAMS_PATH,
        RANDOM_STATE,
        SCALER_PATH,
        TARGET_MULTIPLIER,
    )
    from data_preprocessing import (
        build_iqr_outlier_report,
        fit_scaler,
        impute_numeric_values,
        load_housing_data,
        save_scaler,
        split_housing_data,
    )
    from model_evaluation import (
        empirical_error_band,
        permutation_importance_ann,
        save_evaluation_outputs,
    )


SEARCH_SPACE = [
    {
        "hidden_units": 64,
        "dropout_rate": 0.10,
        "learning_rate": 0.001,
        "batch_size": 128,
    },
    {
        "hidden_units": 128,
        "dropout_rate": 0.20,
        "learning_rate": 0.001,
        "batch_size": 128,
    },
    {
        "hidden_units": 96,
        "dropout_rate": 0.10,
        "learning_rate": 0.0005,
        "batch_size": 256,
    },
]


def set_reproducible_seeds(seed: int = RANDOM_STATE) -> None:
    """Set Python, NumPy, and TensorFlow random seeds."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_ann_regressor(
    input_dim: int,
    hidden_units: int = 128,
    dropout_rate: float = 0.20,
    learning_rate: float = 0.001,
):
    """Build a compact ANN for tabular regression."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(hidden_units // 2, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def tune_model(
    x_train: np.ndarray,
    y_train: pd.Series,
    x_validation: np.ndarray,
    y_validation: pd.Series,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Run the small, transparent hyperparameter search used in the notebook."""
    import tensorflow as tf
    from sklearn.metrics import mean_squared_error

    best_score = float("inf")
    best_params: dict[str, float | int] | None = None
    rows: list[dict[str, float | int]] = []

    for params in SEARCH_SPACE:
        model = build_ann_regressor(
            input_dim=x_train.shape[1],
            hidden_units=int(params["hidden_units"]),
            dropout_rate=float(params["dropout_rate"]),
            learning_rate=float(params["learning_rate"]),
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_mae",
                mode="min",
                patience=4,
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=40,
            batch_size=int(params["batch_size"]),
            verbose=0,
            callbacks=callbacks,
        )
        predictions = model.predict(x_validation, verbose=0).ravel()
        validation_rmse = float(
            np.sqrt(mean_squared_error(y_validation, predictions))
        )

        row = dict(params)
        row["val_rmse"] = validation_rmse
        row["epochs_ran"] = len(history.history["loss"])
        rows.append(row)

        if validation_rmse < best_score:
            best_score = validation_rmse
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("Hyperparameter tuning did not produce a model.")
    return best_params, pd.DataFrame(rows).sort_values("val_rmse")


def train_project(data_path: str | Path = DATA_PATH) -> dict[str, float]:
    """Execute the complete training workflow and save reproducible artifacts."""
    import tensorflow as tf

    set_reproducible_seeds()
    data = load_housing_data(data_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    build_iqr_outlier_report(data).to_csv(
        OUTPUT_DIR / "outlier_report.csv", index=False
    )

    (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
    ) = split_housing_data(data)

    x_train, x_validation, x_test, imputation_values = impute_numeric_values(
        x_train, x_validation, x_test
    )
    scaler = fit_scaler(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    best_params, tuning_results = tune_model(
        x_train_scaled, y_train, x_validation_scaled, y_validation
    )
    tuning_results.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

    model = build_ann_regressor(
        input_dim=len(FEATURE_COLUMNS),
        hidden_units=int(best_params["hidden_units"]),
        dropout_rate=float(best_params["dropout_rate"]),
        learning_rate=float(best_params["learning_rate"]),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            mode="min",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        x_train_scaled,
        y_train,
        validation_data=(x_validation_scaled, y_validation),
        epochs=100,
        batch_size=int(best_params["batch_size"]),
        verbose=1,
        callbacks=callbacks,
    )
    pd.DataFrame(history.history).to_csv(
        OUTPUT_DIR / "training_history.csv", index=False
    )

    test_predictions = model.predict(x_test_scaled, verbose=0).ravel()
    importance = permutation_importance_ann(
        model, scaler, x_test, y_test, repeats=10
    )
    metrics = save_evaluation_outputs(
        y_test.to_numpy(),
        test_predictions,
        OUTPUT_DIR,
        importance=importance,
    )
    interval = empirical_error_band(y_test.to_numpy(), test_predictions, 0.80)

    model.save(MODEL_PATH)
    save_scaler(scaler, SCALER_PATH)
    with PARAMS_PATH.open("w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2)

    q1, median, q3 = [
        float(value) for value in y_train.quantile([0.25, 0.50, 0.75])
    ]
    feature_stats = {}
    for column in FEATURE_COLUMNS:
        series = x_train[column]
        feature_stats[column] = {
            "min": float(series.min()),
            "p01": float(series.quantile(0.01)),
            "p05": float(series.quantile(0.05)),
            "median": float(series.median()),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
        }

    metadata = {
        "project_name": "House Price Prediction using Artificial Neural Networks",
        "dataset_name": "California Housing",
        "dataset_rows": int(len(data)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "SalePrice",
        "target_unit": "USD_100000",
        "target_multiplier": TARGET_MULTIPLIER,
        "model_scope": (
            "Median block-group house value estimation; not an appraisal "
            "of an individual property."
        ),
        "split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
            "random_state": RANDOM_STATE,
        },
        "best_hyperparameters": best_params,
        "metrics_target_units": metrics,
        "prediction_interval": {
            "method": "Empirical 80% absolute-error band on held-out test data",
            "half_width_target_units": interval,
            "half_width_usd": interval * TARGET_MULTIPLIER,
            "warning": (
                "This is an empirical error band, not a calibrated "
                "probabilistic confidence interval."
            ),
        },
        "category_thresholds_target_units": {
            "budget_upper": q1,
            "mid_range_upper": median,
            "premium_upper": q3,
        },
        "feature_statistics": feature_stats,
        "imputation_values": imputation_values,
        "outlier_policy": (
            "Retain observations and report IQR diagnostics; no blind row deletion."
        ),
        "target_ceiling_note": (
            "The source target is capped at approximately $500,001."
        ),
    }
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to house_prices.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = train_project(arguments.data_path)
    print(json.dumps(result, indent=2))
