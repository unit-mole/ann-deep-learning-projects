"""Feature-order safeguards and optional exploratory features.

The deployed ANN intentionally uses the eight original California Housing
predictors so that inference remains compatible with the supplied model.
Optional ratios below are for exploratory analysis or future retraining only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .config import FEATURE_COLUMNS
except ImportError:
    from config import FEATURE_COLUMNS


def select_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return features in the exact order expected by the ANN."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    output = frame[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def add_optional_exploratory_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable ratios without changing the deployed model inputs."""
    output = select_model_features(frame)

    safe_bedrooms = output["AveBedrms"].replace(0, np.nan)
    safe_occupancy = output["AveOccup"].replace(0, np.nan)

    output["RoomsPerBedroom"] = output["AveRooms"] / safe_bedrooms
    output["IncomePerOccupant"] = output["MedInc"] / safe_occupancy
    output["EstimatedHouseholds"] = output["Population"] / safe_occupancy

    return output.replace([np.inf, -np.inf], np.nan)
