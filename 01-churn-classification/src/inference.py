from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR
from src.preprocessing import transform_legacy
from src.validation import validate_input_frame

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def _load_keras_model(path: Path) -> Any:
    # Lazy import keeps validation-only tests lightweight.
    from keras.models import load_model

    return load_model(path, compile=False)


def _risk_band(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Moderate"
    return "High"


class ChurnPredictor:
    """Load modern or legacy artifacts and generate churn probabilities."""

    def __init__(self, model_dir: Path | str = MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.metadata = self._load_metadata()
        self.default_threshold = float(
            self.metadata.get("default_threshold", 0.5)
        )

        modern_model = self.model_dir / "model.keras"
        modern_preprocessor = self.model_dir / "preprocessor.joblib"

        if modern_model.exists() and modern_preprocessor.exists():
            self.mode = "modern_pipeline"
            self.model = _load_keras_model(modern_model)
            self.preprocessor = joblib.load(modern_preprocessor)
            self.label_encoder_gender = None
            self.onehot_encoder_geo = None
            self.scaler = None
        else:
            self.mode = "legacy_notebook_artifacts"
            self.model = _load_keras_model(self.model_dir / "model.h5")
            with (self.model_dir / "label_encoder_gender.pkl").open("rb") as file:
                self.label_encoder_gender = pickle.load(file)
            with (self.model_dir / "onehot_encoder_geo.pkl").open("rb") as file:
                self.onehot_encoder_geo = pickle.load(file)
            with (self.model_dir / "scaler.pkl").open("rb") as file:
                self.scaler = pickle.load(file)
            self.preprocessor = None

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            return {"default_threshold": 0.5}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def predict(
        self,
        frame: pd.DataFrame,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        validated = validate_input_frame(frame)
        decision_threshold = (
            self.default_threshold if threshold is None else float(threshold)
        )

        if not 0.0 < decision_threshold < 1.0:
            raise ValueError("The decision threshold must be between 0 and 1.")

        if self.mode == "modern_pipeline":
            processed = self.preprocessor.transform(validated)
        else:
            processed = transform_legacy(
                validated,
                self.label_encoder_gender,
                self.onehot_encoder_geo,
                self.scaler,
            )

        probabilities = np.asarray(
            self.model.predict(processed, verbose=0)
        ).reshape(-1)
        predictions = (probabilities >= decision_threshold).astype(int)

        result = validated.copy()
        result["ChurnProbability"] = probabilities
        result["PredictedChurn"] = predictions
        result["PredictionLabel"] = np.where(
            predictions == 1,
            "Likely to churn",
            "Not likely to churn",
        )
        result["RiskBand"] = [_risk_band(value) for value in probabilities]
        result["DecisionThreshold"] = decision_threshold
        return result
