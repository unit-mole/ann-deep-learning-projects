from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    INTEGER_LIKE_FEATURES,
    NON_NEGATIVE_FEATURES,
    NUMERIC_FEATURES,
    PROPORTION_FEATURES,
    SCORE_FEATURES,
)
from src.feature_engineering import add_derived_features, infer_legacy_customer_segment


class SchemaValidationError(ValueError):
    """Raised when inference data cannot be aligned to the model schema."""


def numeric_defaults_from_scaler(scaler: StandardScaler) -> dict[str, float]:
    means = getattr(scaler, "mean_", np.zeros(len(NUMERIC_FEATURES)))
    names = list(getattr(scaler, "feature_names_in_", NUMERIC_FEATURES))
    mapping = {name: float(value) for name, value in zip(names, means)}
    return {name: mapping.get(name, 0.0) for name in NUMERIC_FEATURES}


def prepare_inference_frame(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    label_encoders: dict[str, LabelEncoder],
    categorical_fallbacks: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if frame.empty:
        raise SchemaValidationError("The input file does not contain any customer rows.")

    fallbacks = categorical_fallbacks or {}
    warnings: list[str] = []
    df = add_derived_features(frame.copy())
    if "customer_id" not in df:
        df.insert(0, "customer_id", [f"ROW_{i + 1:05d}" for i in range(len(df))])

    if "customer_segment_name" not in df:
        df["customer_segment_name"] = infer_legacy_customer_segment(df)
        warnings.append(
            "customer_segment_name was absent, so the documented legacy-segment heuristic was used."
        )

    defaults = numeric_defaults_from_scaler(scaler)
    for column in NUMERIC_FEATURES:
        if column not in df:
            df[column] = defaults[column]
            warnings.append(f"{column} was missing and was filled with its training mean.")
        converted = pd.to_numeric(df[column], errors="coerce")
        bad_count = int(converted.isna().sum())
        if bad_count:
            warnings.append(f"{column}: {bad_count} invalid or missing value(s) were filled with the training mean.")
        df[column] = converted.fillna(defaults[column])
        if column in NON_NEGATIVE_FEATURES:
            df[column] = df[column].clip(lower=0)
        if column in PROPORTION_FEATURES:
            df[column] = df[column].clip(0, 1)
        if column in SCORE_FEATURES:
            df[column] = df[column].clip(0, 100)
        if column in INTEGER_LIKE_FEATURES:
            df[column] = df[column].round()

    for column in CATEGORICAL_FEATURES:
        encoder = label_encoders[column]
        allowed = {str(value) for value in encoder.classes_}
        fallback = str(fallbacks.get(column, next(iter(allowed))))
        if fallback not in allowed:
            fallback = str(encoder.classes_[0])
        if column not in df:
            df[column] = fallback
            warnings.append(f"{column} was missing and was filled with {fallback!r}.")
        df[column] = df[column].astype("string").fillna(fallback).astype(str)
        unknown = ~df[column].isin(allowed)
        if unknown.any():
            values = sorted(df.loc[unknown, column].unique().tolist())[:5]
            warnings.append(
                f"{column}: {int(unknown.sum())} unknown value(s) {values} were replaced with {fallback!r}."
            )
            df.loc[unknown, column] = fallback

    return df, warnings


def encode_model_inputs(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    label_encoders: dict[str, LabelEncoder],
) -> dict[str, np.ndarray]:
    numeric = scaler.transform(frame[NUMERIC_FEATURES]).astype("float32")
    inputs: dict[str, np.ndarray] = {"numeric_input": numeric}
    for column in CATEGORICAL_FEATURES:
        encoded = label_encoders[column].transform(frame[column].astype(str))
        inputs[f"{column}_input"] = encoded.astype("int32").reshape(-1, 1)
    return inputs


def fit_preprocessors(
    train_frame: pd.DataFrame,
) -> tuple[StandardScaler, dict[str, LabelEncoder], dict[str, int]]:
    scaler = StandardScaler().fit(train_frame[NUMERIC_FEATURES])
    label_encoders: dict[str, LabelEncoder] = {}
    vocab_sizes: dict[str, int] = {}
    for column in CATEGORICAL_FEATURES:
        encoder = LabelEncoder()
        values = train_frame[column].astype(str).tolist() + ["__UNKNOWN__"]
        encoder.fit(values)
        label_encoders[column] = encoder
        vocab_sizes[column] = len(encoder.classes_)
    return scaler, label_encoders, vocab_sizes


def transform_with_unknown(
    frame: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder],
) -> dict[str, np.ndarray]:
    inputs: dict[str, np.ndarray] = {}
    for column in CATEGORICAL_FEATURES:
        encoder = label_encoders[column]
        allowed = set(encoder.classes_)
        values = frame[column].astype(str).where(frame[column].astype(str).isin(allowed), "__UNKNOWN__")
        inputs[f"{column}_input"] = encoder.transform(values).astype("int32").reshape(-1, 1)
    return inputs
