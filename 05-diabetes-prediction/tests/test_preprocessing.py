import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.data_preprocessing import build_preprocessor
from src.feature_engineering import (
    calculate_reference_ranges,
    find_out_of_range_warnings,
    prepare_feature_frame,
)


def test_zero_markers_become_missing_but_pregnancies_stay_zero():
    row = pd.DataFrame(
        [
            {
                "Pregnancies": 0,
                "Glucose": 0,
                "BloodPressure": 0,
                "SkinThickness": 0,
                "Insulin": 0,
                "BMI": 0,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 30,
            }
        ]
    )
    prepared = prepare_feature_frame(row)
    assert prepared.loc[0, "Pregnancies"] == 0
    assert prepared[
        ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    ].isna().all(axis=None)


def test_preprocessor_returns_finite_values():
    data = pd.DataFrame(
        [
            [0, 0, 0, 0, 0, 0, 0.3, 25],
            [2, 120, 70, 25, 80, 30, 0.5, 35],
            [5, 160, 80, 35, 120, 38, 0.8, 50],
        ],
        columns=FEATURE_COLUMNS,
    )
    transformed = build_preprocessor().fit_transform(prepare_feature_frame(data))
    assert np.isfinite(transformed).all()


def test_missing_column_has_clear_error():
    data = pd.DataFrame(
        [{column: 1 for column in FEATURE_COLUMNS if column != "Glucose"}]
    )
    with pytest.raises(ValueError, match="Missing required columns: Glucose"):
        prepare_feature_frame(data)


def test_invalid_text_is_not_silently_imputed():
    row = {column: 1 for column in FEATURE_COLUMNS}
    row["Glucose"] = "not-a-number"
    with pytest.raises(ValueError, match="Glucose: non-numeric"):
        prepare_feature_frame(pd.DataFrame([row]))


def test_negative_values_are_rejected():
    row = {column: 1 for column in FEATURE_COLUMNS}
    row["BMI"] = -1
    with pytest.raises(ValueError, match="BMI: negative"):
        prepare_feature_frame(pd.DataFrame([row]))


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="no data rows"):
        prepare_feature_frame(pd.DataFrame(columns=FEATURE_COLUMNS))


def test_out_of_range_values_generate_non_blocking_warnings():
    reference = pd.DataFrame(
        [
            [0, 80, 60, 20, 40, 22, 0.2, 21],
            [5, 180, 90, 45, 300, 45, 1.2, 70],
        ],
        columns=FEATURE_COLUMNS,
    )
    ranges = calculate_reference_ranges(reference)
    candidate = reference.iloc[[0]].copy()
    candidate.loc[candidate.index[0], "Glucose"] = 250

    warnings = find_out_of_range_warnings(candidate, ranges)

    assert any("Glucose" in warning for warning in warnings)
    assert candidate.loc[candidate.index[0], "Glucose"] == 250


def test_zero_missing_markers_do_not_trigger_range_warnings():
    reference = pd.DataFrame(
        [
            [0, 80, 60, 20, 40, 22, 0.2, 21],
            [5, 180, 90, 45, 300, 45, 1.2, 70],
        ],
        columns=FEATURE_COLUMNS,
    )
    ranges = calculate_reference_ranges(reference)
    candidate = pd.DataFrame(
        [[0, 0, 0, 0, 0, 0, 0.5, 35]], columns=FEATURE_COLUMNS
    )

    assert find_out_of_range_warnings(candidate, ranges) == []
