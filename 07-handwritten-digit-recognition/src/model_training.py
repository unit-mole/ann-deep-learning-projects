"""Train the tuned dense ANN used by the project."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.data_preprocessing import FLATTENED_SIZE, prepare_mnist

DEFAULT_PARAMS = {
    'hidden_units': 384,
    'dropout_rate': 0.20,
    'learning_rate': 0.0005,
    'batch_size': 256,
}

SEARCH_SPACE = [
    {'hidden_units': 256, 'dropout_rate': 0.20, 'learning_rate': 0.0010, 'batch_size': 128},
    {'hidden_units': 512, 'dropout_rate': 0.30, 'learning_rate': 0.0010, 'batch_size': 128},
    DEFAULT_PARAMS,
]


def set_reproducibility(seed: int = 42) -> None:
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def build_ann(
    input_dim: int = FLATTENED_SIZE,
    hidden_units: int = 384,
    dropout_rate: float = 0.20,
    learning_rate: float = 0.0005,
) -> Any:
    """Create the 784 → hidden → hidden/2 → 10 softmax ANN."""
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name='pixel_vector'),
        tf.keras.layers.Dense(hidden_units, activation='relu', name='dense_features'),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_1'),
        tf.keras.layers.Dense(hidden_units // 2, activation='relu', name='dense_compression'),
        tf.keras.layers.Dropout(dropout_rate, name='dropout_2'),
        tf.keras.layers.Dense(10, activation='softmax', name='digit_probabilities'),
    ], name='mnist_dense_ann')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def tune_hyperparameters(data: Any, epochs: int = 10) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Evaluate the original three-candidate search space on the validation split."""
    import tensorflow as tf

    results: list[dict[str, float]] = []
    for params in SEARCH_SPACE:
        model = build_ann(
            hidden_units=int(params['hidden_units']),
            dropout_rate=float(params['dropout_rate']),
            learning_rate=float(params['learning_rate']),
        )
        history = model.fit(
            data.x_train,
            data.y_train_one_hot,
            validation_data=(data.x_validation, data.y_validation_one_hot),
            epochs=epochs,
            batch_size=int(params['batch_size']),
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', mode='max', patience=2, restore_best_weights=True
            )],
        )
        val_loss, val_accuracy = model.evaluate(
            data.x_validation, data.y_validation_one_hot, verbose=0
        )
        results.append({
            **params,
            'val_accuracy': float(val_accuracy),
            'val_loss': float(val_loss),
            'epochs_ran': len(history.history['loss']),
        })

    best = max(results, key=lambda row: row['val_accuracy'])
    best_params = {key: best[key] for key in DEFAULT_PARAMS}
    return best_params, results


def train_model(
    *,
    output_dir: Path,
    epochs: int = 30,
    tune: bool = False,
    seed: int = 42,
) -> tuple[Any, dict[str, list[float]], dict[str, float]]:
    """Train a fresh final model and save the model, history, and selected parameters."""
    import tensorflow as tf

    set_reproducibility(seed)
    data = prepare_mnist(random_seed=seed)
    params = DEFAULT_PARAMS.copy()
    tuning_results = None
    if tune:
        params, tuning_results = tune_hyperparameters(data)

    # Rebuild from scratch after selection; do not continue training a candidate model.
    model = build_ann(
        hidden_units=int(params['hidden_units']),
        dropout_rate=float(params['dropout_rate']),
        learning_rate=float(params['learning_rate']),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]
    history = model.fit(
        data.x_train,
        data.y_train_one_hot,
        validation_data=(data.x_validation, data.y_validation_one_hot),
        epochs=epochs,
        batch_size=int(params['batch_size']),
        callbacks=callbacks,
        verbose=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / 'digit_recognition_model.keras')
    (output_dir / 'best_params.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
    (output_dir / 'training_history.json').write_text(
        json.dumps(history.history, indent=2), encoding='utf-8'
    )
    if tuning_results is not None:
        (output_dir / 'tuning_results.json').write_text(
            json.dumps(tuning_results, indent=2), encoding='utf-8'
        )
    return model, history.history, params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--tune', action='store_true', help='Repeat the original three-candidate search.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('models'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(output_dir=args.output_dir, epochs=args.epochs, tune=args.tune, seed=args.seed)


if __name__ == '__main__':
    main()
