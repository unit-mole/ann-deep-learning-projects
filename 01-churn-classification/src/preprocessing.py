from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline used by the improved training script."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_legacy(
    frame: pd.DataFrame,
    label_encoder_gender: Any,
    onehot_encoder_geo: Any,
    scaler: Any,
) -> np.ndarray:
    """Apply the exact preprocessing used by the original notebook artifacts."""
    working = frame.copy()

    working["Gender"] = label_encoder_gender.transform(
        working["Gender"].astype(str)
    )

    geography = onehot_encoder_geo.transform(working[["Geography"]])
    geography_frame = pd.DataFrame(
        geography,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
        index=working.index,
    )

    processed = pd.concat(
        [working.drop(columns=["Geography"]), geography_frame],
        axis=1,
    )

    expected_columns = list(getattr(scaler, "feature_names_in_", processed.columns))
    processed = processed.reindex(columns=expected_columns, fill_value=0)

    return scaler.transform(processed)
