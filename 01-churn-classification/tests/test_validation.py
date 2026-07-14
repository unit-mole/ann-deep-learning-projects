import pandas as pd
import pytest

from src.validation import InputValidationError, validate_input_frame


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Male",
                "Age": 40,
                "Tenure": 5,
                "Balance": 75000.0,
                "NumOfProducts": 2,
                "HasCrCard": "Yes",
                "IsActiveMember": True,
                "EstimatedSalary": 100000.0,
            }
        ]
    )


def test_valid_frame_is_normalized() -> None:
    validated = validate_input_frame(valid_frame())

    assert validated.loc[0, "HasCrCard"] == 1
    assert validated.loc[0, "IsActiveMember"] == 1
    assert validated.loc[0, "Gender"] == "Male"


def test_missing_column_raises_clear_error() -> None:
    frame = valid_frame().drop(columns=["Age"])

    with pytest.raises(InputValidationError, match="Age"):
        validate_input_frame(frame)


def test_out_of_range_age_is_rejected() -> None:
    frame = valid_frame()
    frame.loc[0, "Age"] = 140

    with pytest.raises(InputValidationError, match="Age"):
        validate_input_frame(frame)


def test_unsupported_geography_is_rejected() -> None:
    frame = valid_frame()
    frame.loc[0, "Geography"] = "Canada"

    with pytest.raises(InputValidationError, match="Canada"):
        validate_input_frame(frame)
