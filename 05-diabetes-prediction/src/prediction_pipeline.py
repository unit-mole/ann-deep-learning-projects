"""Artifact loading and reusable single/batch prediction pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import FEATURE_COLUMNS, MODEL_DIR
from .feature_engineering import prepare_feature_frame, validate_feature_columns
from .risk_scoring import (
    categorize_probabilities,
    recommendation_for_category,
)


class DiabetesPredictionPipeline:
    """Load trained artifacts once and score patient records consistently."""

    def __init__(self, model_dir: str | Path = MODEL_DIR) -> None:
        model_dir = Path(model_dir)
        model_path = model_dir / "diabetes_ann.keras"
        preprocessor_path = model_dir / "preprocessor.joblib"
        metadata_path = model_dir / "model_metadata.json"

        for path in (model_path, preprocessor_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required artifact not found: {path}. Run `python -m src.model_training`."
                )

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.preprocessor = joblib.load(preprocessor_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.classification_threshold = float(
            self.metadata["classification_threshold"]
        )

    def predict_probabilities(self, data: pd.DataFrame) -> np.ndarray:
        """Return diabetes risk probabilities for validated raw input records."""
        features = prepare_feature_frame(data)
        transformed = self.preprocessor.transform(features)
        probabilities = self.model.predict(transformed, verbose=0).ravel()
        return np.clip(probabilities.astype(float), 0.0, 1.0)

    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return inputs plus probability, band, flag, and interpretation."""
        # Validate the schema before selecting columns so uploaded files receive a
        # clear ValueError instead of an unhandled pandas KeyError.
        validate_feature_columns(data)
        features = data.loc[:, FEATURE_COLUMNS].copy()
        probabilities = self.predict_probabilities(features)
        categories = categorize_probabilities(probabilities)

        result = features.reset_index(drop=True)
        result["DiabetesRiskProbability"] = probabilities
        result["RiskCategory"] = categories.to_numpy()
        result["ScreeningFlag"] = np.where(
            probabilities >= self.classification_threshold,
            "Elevated screening flag",
            "No elevated screening flag",
        )
        result["Interpretation"] = result["RiskCategory"].map(
            recommendation_for_category
        )
        return result
