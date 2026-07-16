from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, MODEL_DIR
from src.prediction_pipeline import PredictionPipeline


def test_sample_input_has_required_columns():
    pipeline = PredictionPipeline(MODEL_DIR)
    sample = pd.read_csv(DATA_DIR / "sample_input.csv")
    assert set(pipeline.feature_columns).issubset(sample.columns)


def test_pipeline_returns_probabilities_between_zero_and_one():
    pipeline = PredictionPipeline(MODEL_DIR)
    sample = pd.read_csv(DATA_DIR / "sample_input.csv").head(2)
    probabilities, _ = pipeline.predict_proba(sample)
    assert len(probabilities) == 2
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_unseen_category_does_not_crash():
    pipeline = PredictionPipeline(MODEL_DIR)
    sample = pd.read_csv(DATA_DIR / "sample_input.csv").head(1)
    sample.loc[0, "region"] = "New_Region_Not_In_Training"
    _, unknown_counts = pipeline.predict_proba(sample)
    assert unknown_counts["region"] == 1
