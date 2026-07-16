"""Lightweight project and inference smoke tests."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    SAMPLE_INPUT_PATH,
    SCALER_PATH,
)
from src.feature_engineering import select_model_features


def test_required_artifacts_exist() -> None:
    for path in [MODEL_PATH, SCALER_PATH, METADATA_PATH, SAMPLE_INPUT_PATH]:
        assert path.exists(), f"Missing artifact: {path}"


def test_sample_schema_and_feature_order() -> None:
    sample = pd.read_csv(SAMPLE_INPUT_PATH)
    selected = select_model_features(sample)
    assert selected.columns.tolist() == FEATURE_COLUMNS
    assert not selected.empty


def test_scaler_matches_model_features() -> None:
    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)
    assert list(scaler.feature_names_in_) == FEATURE_COLUMNS
    assert int(scaler.n_features_in_) == len(FEATURE_COLUMNS)


def test_metadata_is_consistent() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["metrics_target_units"]["r2"] > 0.75
    assert metadata["target_multiplier"] == 100_000


def test_tensorflow_model_smoke_prediction() -> None:
    import tensorflow as tf

    sample = pd.read_csv(SAMPLE_INPUT_PATH).head(2)
    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)
    model = tf.keras.models.load_model(MODEL_PATH)
    predictions = model.predict(scaler.transform(sample[FEATURE_COLUMNS]), verbose=0)
    assert predictions.shape == (2, 1)
