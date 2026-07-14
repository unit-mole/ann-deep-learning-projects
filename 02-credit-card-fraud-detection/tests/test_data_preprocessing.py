import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.data_preprocessing import split_dataset, validate_dataset


def make_dataset(rows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.normal(size=(rows, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    frame["Time"] = np.arange(rows, dtype=float)
    frame["Amount"] = np.abs(frame["Amount"]) * 100
    frame["Class"] = 0
    fraud_indices = np.arange(0, rows, 20)
    frame.loc[fraud_indices, "Class"] = 1
    return frame


def test_validate_dataset_accepts_expected_schema():
    frame = make_dataset()
    validate_dataset(frame, require_target=True)


def test_validate_dataset_rejects_missing_feature():
    frame = make_dataset().drop(columns=["V28"])
    with pytest.raises(ValueError, match="V28"):
        validate_dataset(frame, require_target=True)


def test_validate_dataset_rejects_missing_values():
    frame = make_dataset()
    frame.loc[0, "Amount"] = np.nan
    with pytest.raises(ValueError, match="Missing feature values"):
        validate_dataset(frame, require_target=True)


def test_split_is_stratified_and_complete():
    frame = make_dataset()
    splits = split_dataset(frame)

    assert len(splits.X_train) == 700
    assert len(splits.X_valid) == 150
    assert len(splits.X_test) == 150
    assert (
        len(splits.X_train) + len(splits.X_valid) + len(splits.X_test)
        == len(frame)
    )

    source_rate = frame["Class"].mean()
    assert abs(splits.y_train.mean() - source_rate) < 0.01
    assert abs(splits.y_valid.mean() - source_rate) < 0.01
    assert abs(splits.y_test.mean() - source_rate) < 0.01
