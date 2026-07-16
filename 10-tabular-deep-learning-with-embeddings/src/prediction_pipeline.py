from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_preprocessing import build_model_inputs
from .feature_engineering import (
    business_interpretation,
    probability_bucket,
    validate_feature_columns,
)


class PredictionPipeline:
    """Load saved artifacts and provide safe single or batch inference."""

    def __init__(self, model_directory: Path):
        self.model_directory = Path(model_directory)
        self.metadata = json.loads(
            (self.model_directory / "feature_metadata.json").read_text(encoding="utf-8")
        )
        with open(self.model_directory / "label_encoders.pkl", "rb") as file:
            self.category_encoders = pickle.load(file)
        with open(self.model_directory / "numeric_imputer.pkl", "rb") as file:
            self.numeric_imputer = pickle.load(file)
        with open(self.model_directory / "numeric_scaler.pkl", "rb") as file:
            self.numeric_scaler = pickle.load(file)

        import keras

        self.model = keras.saving.load_model(
            self.model_directory / self.metadata["model_file"],
            compile=False,
        )

    @property
    def feature_columns(self) -> list[str]:
        return self.metadata["categorical_features"] + self.metadata["numerical_features"]

    def _prepare(self, frame: pd.DataFrame):
        validate_feature_columns(frame, self.feature_columns)
        return build_model_inputs(
            frame=frame[self.feature_columns].copy(),
            categorical_features=self.metadata["categorical_features"],
            numerical_features=self.metadata["numerical_features"],
            category_encoders=self.category_encoders,
            numeric_imputer=self.numeric_imputer,
            numeric_scaler=self.numeric_scaler,
            fallback_categories=self.metadata.get("fallback_categories", {}),
        )

    def predict_proba(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
        inputs, unknown_counts = self._prepare(frame)
        probabilities = np.asarray(self.model.predict(inputs, verbose=0)).reshape(-1)
        return probabilities, unknown_counts

    def predict(self, frame: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        threshold = float(threshold if threshold is not None else self.metadata["default_threshold"])
        probabilities, unknown_counts = self.predict_proba(frame)
        predictions = (probabilities >= threshold).astype(int)

        output = frame.copy()
        output["prediction_probability"] = probabilities
        output["predicted_class"] = predictions
        labels = self.metadata["class_labels"]
        output["prediction_label"] = [labels[str(value)] for value in predictions]
        output["business_bucket"] = [probability_bucket(value) for value in probabilities]
        output["interpretation"] = [
            business_interpretation(value, threshold) for value in probabilities
        ]
        output.attrs["unknown_category_counts"] = unknown_counts
        return output
