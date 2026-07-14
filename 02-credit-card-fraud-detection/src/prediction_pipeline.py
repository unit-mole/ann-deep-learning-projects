"""Validated batch and single-row inference for the saved ANN model."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_THRESHOLD_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class PredictionSummary:
    total_transactions: int
    predicted_safe: int
    predicted_fraud: int
    fraud_rate: float
    threshold: float


def _load_keras_model(model_path: Path):
    """Load a Keras model.

    TensorFlow is the supported production backend. A Keras/PyTorch fallback is
    included so the lightweight inference tests can run in environments where
    TensorFlow is unavailable but Keras and PyTorch are already installed.
    """
    try:
        from tensorflow import keras
    except ImportError:
        os.environ.setdefault("KERAS_BACKEND", "torch")
        try:
            import keras
        except ImportError as exc:
            raise RuntimeError(
                "Model inference requires TensorFlow. Install dependencies with "
                "'pip install -r requirements.txt'."
            ) from exc

    return keras.models.load_model(model_path, compile=False)


class FraudPredictionPipeline:
    """Load preprocessing artifacts once and score validated transactions."""

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
        threshold_path: str | Path = DEFAULT_THRESHOLD_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.schema_path = Path(schema_path)
        self.threshold_path = Path(threshold_path)

        for artifact_path in (
            self.model_path,
            self.scaler_path,
            self.schema_path,
            self.threshold_path,
        ):
            if not artifact_path.exists():
                raise FileNotFoundError(f"Required artifact not found: {artifact_path}")

        self.model = _load_keras_model(self.model_path)
        with self.scaler_path.open("rb") as file:
            self.scaler = pickle.load(file)

        self.schema: dict[str, Any] = json.loads(
            self.schema_path.read_text(encoding="utf-8")
        )
        self.threshold_config: dict[str, Any] = json.loads(
            self.threshold_path.read_text(encoding="utf-8")
        )
        self.required_features: list[str] = self.schema.get(
            "required_features", FEATURE_COLUMNS
        )
        self.default_threshold = float(
            self.threshold_config.get("default_threshold", 0.5)
        )

        scaler_features = getattr(self.scaler, "feature_names_in_", None)
        if scaler_features is not None:
            scaler_feature_list = [str(value) for value in scaler_features]
            if scaler_feature_list != self.required_features:
                raise ValueError(
                    "Scaler feature order does not match feature_schema.json."
                )

    def validate_input(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Validate and return model features in the trained order."""
        if dataframe.empty:
            raise ValueError("No transaction rows were supplied.")

        duplicated_columns = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()
        if duplicated_columns:
            raise ValueError(
                "Duplicate column name(s): " + ", ".join(duplicated_columns)
            )

        missing_columns = [
            column
            for column in self.required_features
            if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                "Missing required feature column(s): "
                + ", ".join(missing_columns)
            )

        features = dataframe.loc[:, self.required_features].copy()
        conversion_failures: list[str] = []
        for column in self.required_features:
            converted = pd.to_numeric(features[column], errors="coerce")
            if converted.isna().any() and not features[column].isna().all():
                conversion_failures.append(column)
            features[column] = converted

        if conversion_failures:
            raise TypeError(
                "Non-numeric values found in: "
                + ", ".join(conversion_failures)
            )
        if features.isna().any().any():
            missing_counts = features.isna().sum()
            details = {
                column: int(count)
                for column, count in missing_counts.items()
                if count > 0
            }
            raise ValueError(f"Missing feature values detected: {details}")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Infinite feature values are not supported.")

        return features

    def predict_probabilities(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities for each row."""
        features = self.validate_input(dataframe)
        transformed = self.scaler.transform(features)
        probabilities = self.model.predict(
            transformed,
            batch_size=min(4096, max(1, len(features))),
            verbose=0,
        )
        return np.asarray(probabilities, dtype=float).ravel()

    @staticmethod
    def _risk_band(probability: float, threshold: float) -> str:
        if probability >= threshold:
            return "High"
        if probability >= max(0.01, threshold * 0.25):
            return "Review"
        return "Low"

    def predict(
        self,
        dataframe: pd.DataFrame,
        *,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Return original rows plus probability, label, and risk band."""
        selected_threshold = (
            self.default_threshold if threshold is None else float(threshold)
        )
        if not 0 <= selected_threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")

        probabilities = self.predict_probabilities(dataframe)
        predictions = (probabilities >= selected_threshold).astype(int)

        result = dataframe.copy().reset_index(drop=True)
        result["fraud_probability"] = probabilities
        result["fraud_probability_pct"] = probabilities * 100
        result["prediction"] = predictions
        result["prediction_label"] = np.where(
            predictions == 1, "FRAUD / RISK", "SAFE"
        )
        result["risk_band"] = [
            self._risk_band(probability, selected_threshold)
            for probability in probabilities
        ]
        result["decision_threshold"] = selected_threshold

        if TARGET_COLUMN in result.columns:
            actual = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
            valid_actual = actual.isin([0, 1])
            correctness = pd.Series(
                pd.NA,
                index=result.index,
                dtype="boolean",
            )
            correctness.loc[valid_actual] = (
                actual.loc[valid_actual].astype(int).to_numpy()
                == result.loc[valid_actual, "prediction"].to_numpy()
            )
            result["prediction_correct"] = correctness

        return result

    def summarize(
        self,
        predictions: pd.DataFrame,
        *,
        threshold: float | None = None,
    ) -> PredictionSummary:
        """Summarize prediction counts."""
        selected_threshold = (
            self.default_threshold if threshold is None else float(threshold)
        )
        fraud_count = int((predictions["prediction"] == 1).sum())
        total = int(len(predictions))
        return PredictionSummary(
            total_transactions=total,
            predicted_safe=total - fraud_count,
            predicted_fraud=fraud_count,
            fraud_rate=(fraud_count / total if total else 0.0),
            threshold=selected_threshold,
        )
