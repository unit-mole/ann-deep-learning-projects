"""Model loading and digit prediction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DigitPrediction:
    predicted_digit: int
    confidence: float
    second_digit: int
    second_confidence: float
    probabilities: np.ndarray


def load_digit_model(model_path: str | Path) -> Any:
    """Load a Keras model without recompiling it for lightweight inference."""
    import tensorflow as tf

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f'Model file not found: {path}')
    return tf.keras.models.load_model(path, compile=False)


def adapt_input_to_model(model: Any, flattened_input: np.ndarray) -> np.ndarray:
    """Adapt a flattened 784-value input to either ANN or CNN Keras input shapes."""
    array = np.asarray(flattened_input, dtype=np.float32)
    if array.shape != (1, 784):
        raise ValueError(f'Expected input shape (1, 784); received {array.shape}.')

    input_shape = tuple(model.input_shape)
    if len(input_shape) == 2 and input_shape[-1] == 784:
        return array
    if len(input_shape) == 4 and input_shape[1:] == (28, 28, 1):
        return array.reshape(1, 28, 28, 1)
    raise ValueError(f'Unsupported model input shape: {input_shape}')


def predict_digit(model: Any, flattened_input: np.ndarray) -> DigitPrediction:
    """Return the most likely digit, runner-up, and all ten class probabilities."""
    model_input = adapt_input_to_model(model, flattened_input)
    probabilities = np.asarray(model.predict(model_input, verbose=0)[0], dtype=float)
    if probabilities.shape != (10,):
        raise ValueError(f'Expected ten class probabilities; received {probabilities.shape}.')

    ranking = np.argsort(probabilities)[::-1]
    top, second = int(ranking[0]), int(ranking[1])
    return DigitPrediction(
        predicted_digit=top,
        confidence=float(probabilities[top]),
        second_digit=second,
        second_confidence=float(probabilities[second]),
        probabilities=probabilities,
    )
