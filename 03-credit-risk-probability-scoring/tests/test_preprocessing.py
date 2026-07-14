import pandas as pd

from src.config import MODEL_DIR
from src.prediction_pipeline import PortablePreprocessor


def test_portable_preprocessor_shape():
    sample = pd.read_csv("data/sample_input.csv")
    preprocessor = PortablePreprocessor.from_json(MODEL_DIR / "preprocessing_schema.json")
    transformed = preprocessor.transform(sample)
    assert transformed.shape == (len(sample), 41)
