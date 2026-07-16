"""MNIST loading and preprocessing for the dense ANN model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
NUM_CLASSES = 10
IMAGE_SIZE = (28, 28)
FLATTENED_SIZE = 784


@dataclass(frozen=True)
class MNISTData:
    """Prepared arrays used for training, validation, and testing."""

    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    y_train_one_hot: np.ndarray
    y_validation_one_hot: np.ndarray
    y_test_one_hot: np.ndarray
    test_images: np.ndarray


def load_raw_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load MNIST through TensorFlow/Keras and cache it in the user's Keras directory."""
    import tensorflow as tf

    return tf.keras.datasets.mnist.load_data()


def normalize_and_flatten(images: np.ndarray) -> np.ndarray:
    """Convert ``(n, 28, 28)`` uint8 images to normalized ``(n, 784)`` float32 vectors."""
    array = np.asarray(images)
    if array.ndim != 3 or array.shape[1:] != IMAGE_SIZE:
        raise ValueError(f"Expected images with shape (n, 28, 28); received {array.shape}.")
    return array.reshape(array.shape[0], FLATTENED_SIZE).astype(np.float32) / 255.0


def prepare_mnist(validation_size: float = 0.10, random_seed: int = RANDOM_SEED) -> MNISTData:
    """Load, normalize, flatten, stratify, and one-hot encode MNIST."""
    import tensorflow as tf

    (train_images, train_labels), (test_images, test_labels) = load_raw_mnist()
    x_full = normalize_and_flatten(train_images)
    x_test = normalize_and_flatten(test_images)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_full,
        train_labels,
        test_size=validation_size,
        stratify=train_labels,
        random_state=random_seed,
    )

    return MNISTData(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=test_labels,
        y_train_one_hot=tf.keras.utils.to_categorical(y_train, NUM_CLASSES),
        y_validation_one_hot=tf.keras.utils.to_categorical(y_validation, NUM_CLASSES),
        y_test_one_hot=tf.keras.utils.to_categorical(test_labels, NUM_CLASSES),
        test_images=test_images,
    )
