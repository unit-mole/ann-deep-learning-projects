from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
import pandas as pd

UNKNOWN_TOKEN = "__UNKNOWN__"


def choose_embedding_dimension(vocabulary_size: int) -> int:
    """Return a compact, explainable embedding size.

    The rule preserves the notebook's practical approach:
    min(50, ceil(sqrt(vocabulary_size)) + 1). It grows sub-linearly, avoiding
    unnecessarily wide embeddings for small categorical vocabularies.
    """
    if vocabulary_size < 2:
        return 1
    return min(50, math.ceil(math.sqrt(vocabulary_size)) + 1)


@dataclass
class SafeCategoryEncoder:
    """Training-only categorical encoder with a reserved OOV index."""

    categories_: list[str] = field(default_factory=list)
    mapping_: dict[str, int] = field(default_factory=dict)
    unknown_index: int = 0

    def fit(self, values: Iterable[object]) -> "SafeCategoryEncoder":
        cleaned = pd.Series(values, dtype="object").fillna(UNKNOWN_TOKEN).astype(str)
        categories = sorted(value for value in cleaned.unique() if value != UNKNOWN_TOKEN)
        self.categories_ = categories
        self.mapping_ = {category: index + 1 for index, category in enumerate(categories)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.categories_) + 1

    def transform(self, values: Iterable[object]) -> np.ndarray:
        cleaned = pd.Series(values, dtype="object").fillna(UNKNOWN_TOKEN).astype(str)
        return cleaned.map(self.mapping_).fillna(self.unknown_index).astype("int32").to_numpy()

    def inverse_transform(self, encoded: Iterable[int]) -> list[str]:
        reverse = {value: key for key, value in self.mapping_.items()}
        return [reverse.get(int(index), UNKNOWN_TOKEN) for index in encoded]


def fit_category_encoders(
    train_frame: pd.DataFrame,
    categorical_features: list[str],
) -> dict[str, SafeCategoryEncoder]:
    """Fit encoders on training data only to prevent validation/test leakage."""
    return {
        column: SafeCategoryEncoder().fit(train_frame[column])
        for column in categorical_features
    }


def transform_category_frame(
    frame: pd.DataFrame,
    encoders: dict[str, object],
    categorical_features: list[str],
    fallback_categories: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Transform categories and support both new and legacy saved encoders.

    New ``SafeCategoryEncoder`` objects map unseen values to index 0. The
    supplied legacy sklearn LabelEncoder artifacts have no OOV index, so unseen
    values are mapped to a documented fallback category and counted as warnings.
    """
    fallback_categories = fallback_categories or {}
    arrays: dict[str, np.ndarray] = {}
    unknown_counts: dict[str, int] = {}

    for column in categorical_features:
        values = frame[column].fillna(UNKNOWN_TOKEN).astype(str)
        encoder = encoders[column]

        if isinstance(encoder, SafeCategoryEncoder):
            encoded = encoder.transform(values)
            unknown_counts[column] = int((encoded == encoder.unknown_index).sum())
        elif hasattr(encoder, "classes_") and hasattr(encoder, "transform"):
            known = set(str(value) for value in encoder.classes_)
            unknown_mask = ~values.isin(known)
            fallback = fallback_categories.get(column, str(encoder.classes_[0]))
            if fallback not in known:
                fallback = str(encoder.classes_[0])
            safe_values = values.mask(unknown_mask, fallback)
            encoded = encoder.transform(safe_values).astype("int32")
            unknown_counts[column] = int(unknown_mask.sum())
        else:
            raise TypeError(f"Unsupported encoder type for '{column}': {type(encoder)!r}")

        arrays[f"{column}_input"] = np.asarray(encoded, dtype="int32")

    return arrays, unknown_counts
