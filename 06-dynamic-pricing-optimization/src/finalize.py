"""Evaluate portable Keras artifacts with the PyTorch backend and create outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from .constants import CLASSIFICATION_TARGET_COLUMN, MODEL_VERSION, TARGET_COLUMN
from .prediction_pipeline import DynamicPricingPipeline


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def finalize(project_root: str | Path | None = None) -> dict:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    model_dir = root / "models"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    context = joblib.load(model_dir / "training_context.joblib")
    test_frame = context["test_frame"].copy()
    pipeline = DynamicPricingPipeline(model_dir)

    true_demand = test_frame[TARGET_COLUMN].to_numpy()
    predicted_demand = pipeline.predict_demand(test_frame)
    cls_probability = pipeline.predict_high_demand_probability(test_frame)
    cls_prediction = (cls_probability >= 0.50).astype(int)
    cls_true = test_frame[CLASSIFICATION_TARGET_COLUMN].to_numpy()

    metrics = {
        "model_version": MODEL_VERSION,
        "data_source": "reproducible synthetic retail pricing data",
        "training_backend": "JAX",
        "validated_inference_backend": "PyTorch",
        "test_rows": int(len(test_frame)),
        "regression": {
            "mae": float(mean_absolute_error(true_demand, predicted_demand)),
            "rmse": float(np.sqrt(mean_squared_error(true_demand, predicted_demand))),
            "r2": float(r2_score(true_demand, predicted_demand)),
            "mape_pct": safe_mape(true_demand, predicted_demand),
        },
        "classification": {
            "threshold": 0.50,
            "high_demand_cutoff_units": float(pipeline.metadata["high_demand_cutoff_units"]),
            "accuracy": float(accuracy_score(cls_true, cls_prediction)),
            "precision": float(precision_score(cls_true, cls_prediction, zero_division=0)),
            "recall": float(recall_score(cls_true, cls_prediction, zero_division=0)),
            "f1": float(f1_score(cls_true, cls_prediction, zero_division=0)),
        },
    }
    (model_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    evaluation_payload = {
        "metrics": metrics,
        "test_frame": test_frame,
        "true_demand": true_demand,
        "predicted_demand": predicted_demand,
        "classification_probability": cls_probability,
        "history_regression": context["history_regression"],
        "history_classification": context["history_classification"],
    }
    joblib.dump(evaluation_payload, model_dir / "evaluation_payload.joblib")

    from .model_evaluation import create_evaluation_outputs

    return create_evaluation_outputs(model_dir, output_dir)


if __name__ == "__main__":
    final_metrics = finalize()
    print(json.dumps(final_metrics, indent=2))
