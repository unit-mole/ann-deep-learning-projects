"""Load saved artifacts and serve consistent single or batch predictions."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .config import (
        FEATURE_COLUMNS,
        METADATA_PATH,
        MODEL_PATH,
        SCALER_PATH,
        TARGET_MULTIPLIER,
    )
    from .feature_engineering import select_model_features
    from .price_prediction import (
        append_business_outputs,
        assign_price_category,
        build_business_interpretation,
        estimated_price_range_usd,
    )
except ImportError:
    from config import (
        FEATURE_COLUMNS,
        METADATA_PATH,
        MODEL_PATH,
        SCALER_PATH,
        TARGET_MULTIPLIER,
    )
    from feature_engineering import select_model_features
    from price_prediction import (
        append_business_outputs,
        assign_price_category,
        build_business_interpretation,
        estimated_price_range_usd,
    )


DISPLAY_NAMES = {
    "MedInc": "median household income",
    "HouseAge": "median house age",
    "AveRooms": "average rooms",
    "AveBedrms": "average bedrooms",
    "Population": "population",
    "AveOccup": "average occupancy",
    "Latitude": "latitude",
    "Longitude": "longitude",
}


class PredictionPipeline:
    """Inference wrapper that keeps preprocessing and model inputs aligned."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        scaler_path: str | Path = SCALER_PATH,
        metadata_path: str | Path = METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.metadata_path = Path(metadata_path)

        for artifact in [self.model_path, self.scaler_path, self.metadata_path]:
            if not artifact.exists():
                raise FileNotFoundError(f"Required artifact not found: {artifact}")

        try:
            import tensorflow as tf
        except ImportError as error:
            raise ImportError(
                "TensorFlow is required for inference. Install project requirements."
            ) from error

        self.model = tf.keras.models.load_model(self.model_path)
        with self.scaler_path.open("rb") as file:
            self.scaler = pickle.load(file)
        with self.metadata_path.open("r", encoding="utf-8") as file:
            self.metadata: dict[str, Any] = json.load(file)

        scaler_features = list(getattr(self.scaler, "feature_names_in_", FEATURE_COLUMNS))
        if scaler_features != FEATURE_COLUMNS:
            raise ValueError(
                "Scaler feature order does not match the configured model feature order."
            )

    @property
    def thresholds(self) -> dict[str, float]:
        """Return target-unit category thresholds."""
        return self.metadata["category_thresholds_target_units"]

    @property
    def interval_half_width(self) -> float:
        """Return empirical 80% error half-width in target units."""
        return float(
            self.metadata["prediction_interval"]["half_width_target_units"]
        )

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate, order, coerce, and impute model inputs."""
        ordered = select_model_features(frame)
        medians = {
            column: self.metadata["feature_statistics"][column]["median"]
            for column in FEATURE_COLUMNS
        }
        ordered = ordered.fillna(medians)

        if ordered.isna().any().any():
            bad = ordered.columns[ordered.isna().any()].tolist()
            raise ValueError(f"Unable to convert values to numeric form for: {bad}")
        return ordered

    def predict_target_units(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict median value in the source dataset's $100,000 units."""
        prepared = self.prepare(frame)
        scaled = self.scaler.transform(prepared)
        return self.model.predict(scaled, verbose=0).ravel()

    def predict_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return input features plus business-friendly prediction columns."""
        prepared = self.prepare(frame)
        predictions = self.predict_target_units(prepared)
        return append_business_outputs(
            prepared,
            predictions,
            self.thresholds,
            self.interval_half_width,
        )

    def local_drivers(self, row: pd.DataFrame, top_n: int = 3) -> list[dict[str, object]]:
        """Estimate local feature contributions versus a median reference profile.

        This is a sensitivity explanation, not a causal attribution or SHAP value.
        """
        prepared = self.prepare(row).iloc[[0]].copy()
        baseline = float(self.predict_target_units(prepared)[0])
        drivers: list[dict[str, object]] = []

        for column in FEATURE_COLUMNS:
            reference = prepared.copy()
            reference[column] = self.metadata["feature_statistics"][column]["median"]
            reference_prediction = float(self.predict_target_units(reference)[0])
            impact_usd = (baseline - reference_prediction) * TARGET_MULTIPLIER
            drivers.append(
                {
                    "feature": column,
                    "display_name": DISPLAY_NAMES[column],
                    "impact_usd": float(impact_usd),
                    "direction": "upward" if impact_usd >= 0 else "downward",
                }
            )

        return sorted(
            drivers, key=lambda item: abs(float(item["impact_usd"])), reverse=True
        )[:top_n]

    def predict_one(self, values: dict[str, float]) -> dict[str, object]:
        """Predict one row and return a complete presentation-ready result."""
        row = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        prediction = float(self.predict_target_units(row)[0])
        prediction_usd = prediction * TARGET_MULTIPLIER
        low, high = estimated_price_range_usd(
            prediction, self.interval_half_width
        )
        category = assign_price_category(prediction, self.thresholds)
        drivers = self.local_drivers(row)

        return {
            "prediction_target_units": prediction,
            "prediction_usd": prediction_usd,
            "estimated_low_usd": low,
            "estimated_high_usd": high,
            "category": category,
            "local_drivers": drivers,
            "interpretation": build_business_interpretation(category, drivers),
        }
