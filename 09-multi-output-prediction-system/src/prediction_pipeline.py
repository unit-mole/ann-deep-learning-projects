from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import add_deployment_features
from .multi_output_scoring import add_business_outputs


class MultiOutputPredictionPipeline:
    def __init__(self, project_root: str | Path | None = None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        model_dir = self.project_root / "models"
        self.preprocessor = joblib.load(model_dir / "preprocessor.joblib")
        self.metadata = json.loads((model_dir / "target_metadata.json").read_text(encoding="utf-8"))
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        from keras.saving import load_model
        self.model = load_model(model_dir / "multi_output_model.keras", compile=False)

    @staticmethod
    def _dense(values):
        return values.toarray() if hasattr(values, "toarray") else values

    def predict(self, raw_df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        features = add_deployment_features(raw_df, self.metadata)
        processed = self._dense(self.preprocessor.transform(features)).astype("float32")
        churn_probability, clv_scaled, engagement_scaled = self.model.predict(processed, verbose=0)
        churn_probability = np.asarray(churn_probability).reshape(-1)
        clv = np.asarray(clv_scaled).reshape(-1) * self.metadata["clv_std"] + self.metadata["clv_mean"]
        engagement = np.asarray(engagement_scaled).reshape(-1) * self.metadata["engagement_std"] + self.metadata["engagement_mean"]
        decision_threshold = float(threshold if threshold is not None else self.metadata["churn_threshold"])
        result = raw_df.reset_index(drop=True).copy()
        result["predicted_churn_probability"] = churn_probability
        result["predicted_churn"] = (churn_probability >= decision_threshold).astype(int)
        result["predicted_clv"] = clv
        result["predicted_engagement"] = np.clip(engagement, 0, 100)
        result["decision_threshold"] = decision_threshold
        return add_business_outputs(result)
