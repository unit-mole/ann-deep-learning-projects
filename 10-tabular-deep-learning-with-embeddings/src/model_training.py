from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .embedding_preprocessing import choose_embedding_dimension


def build_tabular_embedding_model(
    categorical_features: list[str],
    category_encoders: dict[str, object],
    numerical_feature_count: int,
):
    """Build a functional Keras binary classifier for mixed tabular inputs."""
    from keras import Model
    from keras.layers import (
        BatchNormalization,
        Concatenate,
        Dense,
        Dropout,
        Embedding,
        Flatten,
        Input,
    )

    inputs = []
    embedding_vectors = []

    for column in categorical_features:
        encoder = category_encoders[column]
        if hasattr(encoder, "vocab_size"):
            vocabulary_size = int(encoder.vocab_size)
        elif hasattr(encoder, "classes_"):
            vocabulary_size = len(encoder.classes_)
        else:
            raise TypeError(f"Unsupported encoder for {column}")

        # Compact sub-linear rule documented in README.md.
        embedding_dimension = choose_embedding_dimension(vocabulary_size)
        category_input = Input(shape=(1,), name=f"{column}_input", dtype="int32")
        embedded = Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dimension,
            name=f"{column}_embedding",
        )(category_input)
        embedded = Flatten(name=f"{column}_flatten")(embedded)
        inputs.append(category_input)
        embedding_vectors.append(embedded)

    numerical_input = Input(
        shape=(numerical_feature_count,),
        name="numerical_input",
        dtype="float32",
    )
    inputs.append(numerical_input)

    x = Concatenate(name="feature_concat")(embedding_vectors + [numerical_input])
    x = Dense(128, activation="relu", name="dense_128")(x)
    x = BatchNormalization(name="batch_norm_128")(x)
    x = Dropout(0.30, name="dropout_128")(x)
    x = Dense(64, activation="relu", name="dense_64")(x)
    x = BatchNormalization(name="batch_norm_64")(x)
    x = Dropout(0.25, name="dropout_64")(x)
    x = Dense(32, activation="relu", name="dense_32")(x)
    x = Dropout(0.20, name="dropout_32")(x)
    output = Dense(1, activation="sigmoid", name="target_probability")(x)

    return Model(inputs=inputs, outputs=output, name="Tabular_Embedding_ANN")


def compile_binary_model(model):
    """Compile the ANN with binary-classification loss and ranking metrics."""
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def compute_class_weights(target: pd.Series) -> dict[int, float]:
    """Return balanced class weights for an imbalanced binary target."""
    counts = target.value_counts().sort_index()
    total = counts.sum()
    return {
        int(class_id): float(total / (len(counts) * count))
        for class_id, count in counts.items()
    }


def train_embedding_model(
    model,
    train_inputs: dict[str, np.ndarray],
    train_target: np.ndarray,
    validation_inputs: dict[str, np.ndarray],
    validation_target: np.ndarray,
    class_weight: dict[int, float],
    epochs: int = 25,
    batch_size: int = 256,
):
    """Train with early stopping and learning-rate reduction."""
    import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    return model.fit(
        train_inputs,
        train_target,
        validation_data=(validation_inputs, validation_target),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )


def train_random_forest_baseline(
    train_frame: pd.DataFrame,
    train_target: pd.Series,
    categorical_features: list[str],
    numerical_features: list[str],
    seed: int = 42,
) -> Pipeline:
    """Fit a leakage-safe one-hot Random Forest comparison baseline.

    All imputers and the one-hot vocabulary are fitted inside the pipeline using
    training data only. ``handle_unknown='ignore'`` makes validation/test scoring
    robust to categories that were not seen during fitting.
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    numerical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numerical", numerical_pipeline, numerical_features),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(train_frame[categorical_features + numerical_features], train_target)
    return pipeline
