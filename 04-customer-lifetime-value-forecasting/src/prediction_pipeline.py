from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.clv_scoring import SegmentThresholds, assign_clv_segments, business_recommendation
from src.config import ENCODERS_PATH, METADATA_PATH, MODEL_PATH, SCALER_PATH
from src.data_preprocessing import encode_model_inputs, prepare_inference_frame


class PredictionError(RuntimeError):
    """Raised when model artifacts cannot be loaded or used."""


class CLVPredictionPipeline:
    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        scaler_path: str | Path = SCALER_PATH,
        encoders_path: str | Path = ENCODERS_PATH,
        metadata_path: str | Path = METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.encoders_path = Path(encoders_path)
        self.metadata_path = Path(metadata_path)
        try:
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.scaler = pd.read_pickle(self.scaler_path)
            self.label_encoders = pd.read_pickle(self.encoders_path)
        except Exception as exc:
            raise PredictionError(f"Unable to load preprocessing artifacts: {exc}") from exc
        self._model = None
        self.last_warnings: list[str] = []

    @property
    def model(self):
        if self._model is None:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            try:
                import keras

                self._model = keras.models.load_model(self.model_path, compile=False)
            except Exception as exc:
                raise PredictionError(
                    "Unable to load the Keras model. Install the versions in requirements.txt "
                    f"and verify {self.model_path.name} is present. Original error: {exc}"
                ) from exc
        return self._model

    @property
    def category_options(self) -> dict[str, list[str]]:
        return {
            column: [str(value) for value in encoder.classes_]
            for column, encoder in self.label_encoders.items()
        }

    @property
    def numeric_defaults(self) -> dict[str, float]:
        names = list(getattr(self.scaler, "feature_names_in_", []))
        values = list(getattr(self.scaler, "mean_", []))
        return {name: float(value) for name, value in zip(names, values)}

    def _reverse_target_transform(self, values: np.ndarray) -> np.ndarray:
        transform = self.metadata.get("target_transform", "none")
        if transform == "log1p":
            return np.expm1(values)
        return values

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared, warnings = prepare_inference_frame(
            frame,
            scaler=self.scaler,
            label_encoders=self.label_encoders,
            categorical_fallbacks=self.metadata.get("categorical_fallbacks", {}),
        )
        self.last_warnings = warnings
        model_inputs = encode_model_inputs(prepared, self.scaler, self.label_encoders)
        try:
            raw_predictions = self.model.predict(model_inputs, verbose=0)
        except Exception as exc:
            raise PredictionError(f"The model could not score the supplied rows: {exc}") from exc

        if isinstance(raw_predictions, dict):
            clv_raw = raw_predictions["clv_output"]
            retention_raw = raw_predictions["retention_output"]
        else:
            clv_raw, retention_raw = raw_predictions

        clv = np.maximum(self._reverse_target_transform(np.asarray(clv_raw).reshape(-1)), 0.0)
        retention = np.clip(np.asarray(retention_raw).reshape(-1), 0.0, 1.0)
        threshold_values = self.metadata["segmentation"]["thresholds"]
        segment_thresholds = SegmentThresholds.from_mapping(threshold_values)
        segments = assign_clv_segments(clv, segment_thresholds)

        output = frame.reset_index(drop=True).copy()
        if "customer_id" not in output:
            output.insert(0, "customer_id", prepared["customer_id"].to_numpy())
        output["predicted_clv_90d"] = np.round(clv, 2)
        output["predicted_retention_probability"] = np.round(retention, 4)
        output["predicted_retained_90d"] = (retention >= 0.50).astype(int)
        output["customer_value_segment"] = segments.to_numpy()
        output["business_recommendation"] = [
            business_recommendation(seg, probability)
            for seg, probability in zip(segments, retention)
        ]
        return output

    def predict_single(self, record: dict[str, Any]) -> dict[str, Any]:
        result = self.predict_dataframe(pd.DataFrame([record])).iloc[0]
        return result.to_dict()
