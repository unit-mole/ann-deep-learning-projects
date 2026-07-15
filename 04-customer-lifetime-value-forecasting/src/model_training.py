from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    OUTPUTS_DIR,
    SEGMENTATION_FEATURES,
)
from src.data_generation import SyntheticDataConfig, generate_synthetic_transactions
from src.data_preprocessing import fit_preprocessors, transform_with_unknown
from src.feature_engineering import build_customer_feature_table
from src.model_evaluation import classification_metrics, numeric_permutation_importance, regression_metrics, save_metrics

SEED = 42


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def _embedding_dim(cardinality: int) -> int:
    return int(min(50, max(4, round(math.sqrt(cardinality) * 2))))


def build_model(vocab_sizes: dict[str, int]):
    import keras
    from keras import Input, Model
    from keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Embedding, Flatten

    numeric_input = Input(shape=(len(NUMERIC_FEATURES),), name="numeric_input")
    all_inputs = [numeric_input]
    embedded = []
    for column in CATEGORICAL_FEATURES:
        categorical_input = Input(shape=(1,), name=f"{column}_input")
        vector = Embedding(vocab_sizes[column] + 1, _embedding_dim(vocab_sizes[column]), name=f"{column}_embedding")(categorical_input)
        embedded.append(Flatten(name=f"{column}_flatten")(vector))
        all_inputs.append(categorical_input)

    numeric_tower = Dense(128, activation="relu")(numeric_input)
    numeric_tower = BatchNormalization()(numeric_tower)
    numeric_tower = Dropout(0.20)(numeric_tower)
    numeric_tower = Dense(64, activation="relu")(numeric_tower)
    numeric_tower = BatchNormalization()(numeric_tower)
    numeric_tower = Dropout(0.15)(numeric_tower)

    shared = Concatenate()([numeric_tower] + embedded)
    for units, dropout in [(256, 0.25), (128, 0.20), (64, 0.15)]:
        shared = Dense(units, activation="relu")(shared)
        shared = BatchNormalization()(shared)
        shared = Dropout(dropout)(shared)

    # The regression target is log1p transformed; a linear output avoids clipping gradients.
    clv_output = Dense(1, activation="linear", name="clv_output")(shared)
    retention_output = Dense(1, activation="sigmoid", name="retention_output")(shared)
    model = Model(all_inputs, [clv_output, retention_output], name="clv_ann_with_embeddings")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"clv_output": keras.losses.Huber(delta=1.0), "retention_output": "binary_crossentropy"},
        loss_weights={"clv_output": 1.0, "retention_output": 0.5},
        metrics={
            "clv_output": [keras.metrics.MeanAbsoluteError(name="mae_log")],
            "retention_output": [keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")],
        },
    )
    return model


def _apply_segments(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    segment_scaler = StandardScaler().fit(train_df[SEGMENTATION_FEATURES])
    segmenter = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(segment_scaler.transform(train_df[SEGMENTATION_FEATURES]))
    train_clusters = segmenter.predict(segment_scaler.transform(train_df[SEGMENTATION_FEATURES]))
    cluster_value = pd.DataFrame({"cluster": train_clusters, "target": train_df["future_90d_revenue"].to_numpy()}).groupby("cluster")["target"].mean().sort_values()
    # Preserve the supplied artifact's semantic ordering: Segment_1 lowest, then 3, 0, 2 highest.
    semantic_names = ["Segment_1", "Segment_3", "Segment_0", "Segment_2"]
    mapping = {int(cluster): name for cluster, name in zip(cluster_value.index, semantic_names)}
    for frame in (train_df, val_df, test_df):
        clusters = segmenter.predict(segment_scaler.transform(frame[SEGMENTATION_FEATURES]))
        frame["customer_segment_name"] = pd.Series(clusters, index=frame.index).map(mapping).astype(str)
    return segment_scaler, segmenter, mapping


def train_project(n_customers: int = 6000, epochs: int = 60, batch_size: int = 128) -> dict:
    import keras

    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    transactions = generate_synthetic_transactions(SyntheticDataConfig(n_customers=n_customers, seed=SEED))
    model_df = build_customer_feature_table(transactions)

    train_df, temp_df = train_test_split(
        model_df, test_size=0.30, random_state=SEED, stratify=model_df["retained_90d"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["retained_90d"]
    )
    train_df, val_df, test_df = (df.reset_index(drop=True) for df in (train_df, val_df, test_df))

    segment_scaler, segmenter, segment_mapping = _apply_segments(train_df, val_df, test_df)
    scaler, encoders, vocab_sizes = fit_preprocessors(train_df)

    def inputs(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        result = {"numeric_input": scaler.transform(frame[NUMERIC_FEATURES]).astype("float32")}
        result.update(transform_with_unknown(frame, encoders))
        return result

    train_inputs, val_inputs, test_inputs = inputs(train_df), inputs(val_df), inputs(test_df)
    y_train_log = np.log1p(train_df["future_90d_revenue"].to_numpy("float32"))
    y_val_log = np.log1p(val_df["future_90d_revenue"].to_numpy("float32"))
    y_test = test_df["future_90d_revenue"].to_numpy("float32")
    y_train_ret = train_df["retained_90d"].to_numpy("float32")
    y_val_ret = val_df["retained_90d"].to_numpy("float32")
    y_test_ret = test_df["retained_90d"].to_numpy("int32")

    model = build_model(vocab_sizes)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
    ]
    history = model.fit(
        train_inputs,
        {"clv_output": y_train_log, "retention_output": y_train_ret},
        validation_data=(val_inputs, {"clv_output": y_val_log, "retention_output": y_val_ret}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_clv_log, _ = model.predict(val_inputs, verbose=0)
    val_clv = np.maximum(np.expm1(val_clv_log.reshape(-1)), 0)
    positive_val = pd.Series(val_clv[val_clv > 0])
    thresholds = {
        "low_max": float(positive_val.quantile(0.25)),
        "medium_max": float(positive_val.quantile(0.75)),
        "high_max": float(positive_val.quantile(0.90)),
    }

    test_clv_log, test_ret_prob = model.predict(test_inputs, verbose=0)
    test_clv = np.maximum(np.expm1(test_clv_log.reshape(-1)), 0)
    test_ret_prob = test_ret_prob.reshape(-1)
    metrics = {
        "test_customers": int(len(test_df)),
        "regression": regression_metrics(y_test, test_clv),
        "retention": classification_metrics(y_test_ret, test_ret_prob),
        "segmentation_thresholds_usd": thresholds,
    }

    predictions = pd.DataFrame({
        "customer_id": test_df["customer_id"],
        "actual_future_90d_revenue": y_test,
        "predicted_future_90d_revenue": test_clv,
        "actual_retained_90d": y_test_ret,
        "predicted_retention_probability": test_ret_prob,
        "predicted_retained_90d": (test_ret_prob >= 0.50).astype(int),
    })
    model.save(MODELS_DIR / "clv_ann_model.keras")
    pd.to_pickle(scaler, MODELS_DIR / "numeric_scaler.pkl")
    pd.to_pickle(encoders, MODELS_DIR / "label_encoders.pkl")
    pd.to_pickle(segment_scaler, MODELS_DIR / "segment_scaler.pkl")
    pd.to_pickle(segmenter, MODELS_DIR / "customer_segmenter.pkl")
    predictions.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)
    pd.DataFrame(history.history).to_csv(OUTPUTS_DIR / "training_history.csv", index=False)
    save_metrics(metrics, OUTPUTS_DIR / "model_metrics.json")

    importance = numeric_permutation_importance(
        model,
        test_inputs,
        y_test,
        NUMERIC_FEATURES,
        reverse_target_transform=np.expm1,
        repeats=3,
        seed=SEED,
    )
    importance.to_csv(OUTPUTS_DIR / "permutation_feature_importance.csv", index=False)

    metadata = {
        "schema_version": 2,
        "project_name": "Customer Lifetime Value Forecasting with ANN",
        "forecast_horizon_days": 90,
        "prediction_definition": "Future customer revenue during the next 90 days; used as a practical CLV proxy.",
        "model_file": "clv_ann_model.keras",
        "target_transform": "log1p",
        "output_order": ["clv_output", "retention_output"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_options": {column: [str(x) for x in encoder.classes_] for column, encoder in encoders.items()},
        "categorical_fallbacks": {column: "__UNKNOWN__" for column in CATEGORICAL_FEATURES},
        "segmentation": {
            "method": "P25/P75/P90 thresholds learned from positive validation predictions.",
            "currency": "USD",
            "thresholds": thresholds,
        },
        "cluster_mapping": segment_mapping,
        "metrics": metrics,
        "dataset": {"source": "Synthetic transaction generator", "contains_personal_data": False},
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CLV multi-task ANN and save deployment artifacts.")
    parser.add_argument("--customers", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    metrics = train_project(args.customers, args.epochs, args.batch_size)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
