import pickle
from pathlib import Path

import pandas as pd

from src.preprocessing import transform_legacy
from src.validation import validate_input_frame


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def test_legacy_preprocessing_produces_twelve_features() -> None:
    frame = pd.DataFrame(
        [
            {
                "CreditScore": 600,
                "Geography": "France",
                "Gender": "Male",
                "Age": 40,
                "Tenure": 3,
                "Balance": 60000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0,
            }
        ]
    )
    validated = validate_input_frame(frame)

    with (MODEL_DIR / "label_encoder_gender.pkl").open("rb") as file:
        gender_encoder = pickle.load(file)
    with (MODEL_DIR / "onehot_encoder_geo.pkl").open("rb") as file:
        geography_encoder = pickle.load(file)
    with (MODEL_DIR / "scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)

    transformed = transform_legacy(
        validated,
        gender_encoder,
        geography_encoder,
        scaler,
    )

    assert transformed.shape == (1, 12)
