"""ANN model builders and the JAX-backed training stage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "jax")

import joblib
import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .constants import (
    CATEGORICAL_COLUMNS,
    CLASSIFICATION_TARGET_COLUMN,
    MODEL_VERSION,
    NUMERIC_FEATURE_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from .data_preprocessing import clean_pricing_data
from .feature_engineering import add_pricing_features


def build_encoder_maps(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Create stable category-to-index maps with zero reserved for unknown values."""

    return {
        column: {value: idx + 1 for idx, value in enumerate(sorted(frame[column].astype(str).unique()))}
        for column in CATEGORICAL_COLUMNS
    }


def encode_categories(frame: pd.DataFrame, encoder_maps: dict[str, dict[str, int]]) -> dict[str, np.ndarray]:
    return {
        column: frame[column].map(encoder_maps[column]).fillna(0).astype("int32").to_numpy()
        for column in CATEGORICAL_COLUMNS
    }


def build_demand_regressor(n_numeric: int, cardinalities: dict[str, int]) -> keras.Model:
    numeric_input = keras.Input(shape=(n_numeric,), name="numeric_input")
    category_input = keras.Input(shape=(1,), dtype="int32", name="category_input")
    region_input = keras.Input(shape=(1,), dtype="int32", name="region_input")
    channel_input = keras.Input(shape=(1,), dtype="int32", name="channel_input")

    category_embedding = layers.Flatten()(layers.Embedding(cardinalities["category"] + 1, 4)(category_input))
    region_embedding = layers.Flatten()(layers.Embedding(cardinalities["region"] + 1, 3)(region_input))
    channel_embedding = layers.Flatten()(layers.Embedding(cardinalities["channel"] + 1, 2)(channel_input))

    merged = layers.Concatenate()([numeric_input, category_embedding, region_embedding, channel_embedding])
    x = layers.Dense(128, activation="relu")(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.22)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.14)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="linear", name="log_demand_output")(x)

    model = keras.Model(
        inputs=[numeric_input, category_input, region_input, channel_input],
        outputs=output,
        name="dynamic_pricing_demand_ann",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss=keras.losses.Huber(delta=0.7),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_demand_classifier(n_numeric: int, cardinalities: dict[str, int]) -> keras.Model:
    numeric_input = keras.Input(shape=(n_numeric,), name="numeric_input")
    category_input = keras.Input(shape=(1,), dtype="int32", name="category_input")
    region_input = keras.Input(shape=(1,), dtype="int32", name="region_input")
    channel_input = keras.Input(shape=(1,), dtype="int32", name="channel_input")

    category_embedding = layers.Flatten()(layers.Embedding(cardinalities["category"] + 1, 4)(category_input))
    region_embedding = layers.Flatten()(layers.Embedding(cardinalities["region"] + 1, 3)(region_input))
    channel_embedding = layers.Flatten()(layers.Embedding(cardinalities["channel"] + 1, 2)(channel_input))

    merged = layers.Concatenate()([numeric_input, category_embedding, region_embedding, channel_embedding])
    x = layers.Dense(96, activation="relu")(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(48, activation="relu")(x)
    x = layers.Dense(24, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid", name="high_demand_probability")(x)

    model = keras.Model(
        inputs=[numeric_input, category_input, region_input, channel_input],
        outputs=output,
        name="dynamic_pricing_demand_state_ann",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _model_inputs(scaled_numeric: np.ndarray, encoded: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [
        scaled_numeric.astype("float32"),
        encoded["category"],
        encoded["region"],
        encoded["channel"],
    ]


def train_models(
    frame: pd.DataFrame,
    artifact_dir: str | Path,
    epochs_regression: int = 20,
    epochs_classification: int = 5,
    batch_size: int = 256,
    verbose: int = 2,
) -> dict[str, Any]:
    """Train and persist portable Keras models plus preprocessing metadata.

    Model evaluation is intentionally executed in a fresh PyTorch-backed process
    by ``src.finalize``. This avoids backend recompilation overhead and verifies
    that the saved Keras artifacts are portable across supported backends.
    """

    keras.utils.set_random_seed(RANDOM_SEED)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    clean = clean_pricing_data(frame, require_target=True)
    featured = add_pricing_features(clean)
    train_df, temp_df = train_test_split(featured, test_size=0.30, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=RANDOM_SEED)

    high_demand_threshold = float(train_df[TARGET_COLUMN].quantile(0.67))
    for split in (train_df, val_df, test_df):
        split[CLASSIFICATION_TARGET_COLUMN] = (split[TARGET_COLUMN] >= high_demand_threshold).astype(int)

    scaler = StandardScaler()
    x_train_num = scaler.fit_transform(train_df[NUMERIC_FEATURE_COLUMNS])
    x_val_num = scaler.transform(val_df[NUMERIC_FEATURE_COLUMNS])
    encoder_maps = build_encoder_maps(train_df)
    encoded_train = encode_categories(train_df, encoder_maps)
    encoded_val = encode_categories(val_df, encoder_maps)
    cardinalities = {column: len(encoder_maps[column]) for column in CATEGORICAL_COLUMNS}

    regressor = build_demand_regressor(len(NUMERIC_FEATURE_COLUMNS), cardinalities)
    classifier = build_demand_classifier(len(NUMERIC_FEATURE_COLUMNS), cardinalities)

    reg_callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    cls_callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    history_reg = regressor.fit(
        _model_inputs(x_train_num, encoded_train),
        np.log1p(train_df[TARGET_COLUMN].to_numpy()).astype("float32"),
        validation_data=(
            _model_inputs(x_val_num, encoded_val),
            np.log1p(val_df[TARGET_COLUMN].to_numpy()).astype("float32"),
        ),
        epochs=epochs_regression,
        batch_size=batch_size,
        callbacks=reg_callbacks,
        verbose=verbose,
    )
    history_cls = classifier.fit(
        _model_inputs(x_train_num, encoded_train),
        train_df[CLASSIFICATION_TARGET_COLUMN].to_numpy().astype("float32"),
        validation_data=(
            _model_inputs(x_val_num, encoded_val),
            val_df[CLASSIFICATION_TARGET_COLUMN].to_numpy().astype("float32"),
        ),
        epochs=epochs_classification,
        batch_size=batch_size,
        callbacks=cls_callbacks,
        verbose=verbose,
    )

    regressor.save(artifact_dir / "dynamic_pricing_demand_ann.keras")
    classifier.save(artifact_dir / "dynamic_pricing_demand_state_ann.keras")
    joblib.dump(scaler, artifact_dir / "numeric_scaler.joblib")

    # Pricing segments are assigned by transparent rules during inference.
    metadata = {
        "model_version": MODEL_VERSION,
        "training_backend": "jax",
        "inference_backend": "torch",
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "encoder_maps": encoder_maps,
        "high_demand_cutoff_units": high_demand_threshold,
        "segment_method": "transparent business rules",
        "segment_definitions": {
            "0": "Value / Low Velocity",
            "1": "Core / Stable Demand",
            "2": "Premium / High Value",
            "3": "Promotion-Sensitive",
        },
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }
    (artifact_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    context = {
        "test_frame": test_df.reset_index(drop=True),
        "history_regression": history_reg.history,
        "history_classification": history_cls.history,
    }
    joblib.dump(context, artifact_dir / "training_context.joblib")
    return metadata
