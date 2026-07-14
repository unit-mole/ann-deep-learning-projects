from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CLASSIFICATION_THRESHOLD, MODEL_DIR
from .feature_engineering import prepare_feature_frame
from .risk_scoring import add_risk_outputs


class PortablePreprocessor:
    """Version-stable transform using exported training statistics and categories."""

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema

    @classmethod
    def from_json(cls, path: str | Path) -> "PortablePreprocessor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        frame = prepare_feature_frame(data)
        blocks: list[np.ndarray] = []
        numeric_values = []
        for feature in self.schema["numeric_features"]:
            series = pd.to_numeric(frame[feature], errors="coerce")
            series = series.fillna(self.schema["numeric_medians"][feature])
            scale = self.schema["numeric_scales"][feature] or 1.0
            numeric_values.append((series.to_numpy(dtype=float) - self.schema["numeric_means"][feature]) / scale)
        blocks.append(np.column_stack(numeric_values))

        for feature in self.schema["categorical_features"]:
            series = frame[feature].astype("object")
            series = series.where(series.notna(), self.schema["categorical_modes"][feature]).astype(str)
            categories = self.schema["categorical_categories"][feature]
            blocks.append(np.column_stack([(series == category).to_numpy(dtype=float) for category in categories]))
        result = np.column_stack(blocks).astype("float32")
        expected = int(self.schema.get("processed_input_dim", result.shape[1]))
        if result.shape[1] != expected:
            raise ValueError(f"Preprocessing generated {result.shape[1]} columns; expected {expected}.")
        return result


def load_model(model_path: str | Path | None = None):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for scoring. Install dependencies with pip install -r requirements.txt") from exc
    path = Path(model_path) if model_path else MODEL_DIR / "final_credit_risk_ann_model.keras"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path, compile=False)


def load_preprocessor(schema_path: str | Path | None = None) -> PortablePreprocessor:
    path = Path(schema_path) if schema_path else MODEL_DIR / "preprocessing_schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing schema not found: {path}")
    return PortablePreprocessor.from_json(path)


def predict_probabilities(data: pd.DataFrame, model=None, preprocessor=None) -> np.ndarray:
    model = model or load_model()
    preprocessor = preprocessor or load_preprocessor()
    processed = preprocessor.transform(data)
    return np.asarray(model.predict(processed, verbose=0)).reshape(-1)


def score_applicants(
    data: pd.DataFrame,
    model=None,
    preprocessor=None,
    classification_threshold: float = CLASSIFICATION_THRESHOLD,
) -> pd.DataFrame:
    probabilities = predict_probabilities(data, model=model, preprocessor=preprocessor)
    return add_risk_outputs(data, probabilities, classification_threshold=classification_threshold)
