import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS
from src.data_preprocessing import build_preprocessor
from src.feature_engineering import prepare_feature_frame


def test_zero_markers_become_missing_but_pregnancies_stay_zero():
    row = pd.DataFrame([{
        "Pregnancies": 0,
        "Glucose": 0,
        "BloodPressure": 0,
        "SkinThickness": 0,
        "Insulin": 0,
        "BMI": 0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30,
    }])
    prepared = prepare_feature_frame(row)
    assert prepared.loc[0, "Pregnancies"] == 0
    assert prepared[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].isna().all(axis=None)


def test_preprocessor_returns_finite_values():
    data = pd.DataFrame([
        [0, 0, 0, 0, 0, 0, 0.3, 25],
        [2, 120, 70, 25, 80, 30, 0.5, 35],
        [5, 160, 80, 35, 120, 38, 0.8, 50],
    ], columns=FEATURE_COLUMNS)
    transformed = build_preprocessor().fit_transform(prepare_feature_frame(data))
    assert np.isfinite(transformed).all()
