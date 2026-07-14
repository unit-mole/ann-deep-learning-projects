from __future__ import annotations

from typing import Any


def build_ann(input_dimension: int, learning_rate: float = 0.001) -> Any:
    """Build the improved tabular ANN used by src/train.py."""
    from keras import Sequential
    from keras.layers import Dense, Dropout, Input
    from keras.metrics import AUC, BinaryAccuracy, Precision, Recall
    from keras.optimizers import Adam

    model = Sequential(
        [
            Input(shape=(input_dimension,), name="customer_features"),
            Dense(64, activation="relu", name="hidden_64"),
            Dropout(0.20, name="dropout_20pct"),
            Dense(16, activation="relu", name="hidden_16"),
            Dense(1, activation="sigmoid", name="churn_probability"),
        ],
        name="ann_churn_classifier",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            BinaryAccuracy(name="accuracy"),
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="roc_auc"),
            AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model
