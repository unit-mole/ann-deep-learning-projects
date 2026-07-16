from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .constants import MODEL_FEATURES, RAW_FEATURES


@dataclass(frozen=True)
class FeatureMetadata:
    spend_bin_edges: list[float]
    spend_bin_labels: list[str]
    tenure_bin_edges: list[float]
    tenure_bin_labels: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def fit_feature_metadata(train_df: pd.DataFrame) -> FeatureMetadata:
    spend = pd.to_numeric(train_df["monthly_spend"], errors="coerce")
    spend = spend.fillna(spend.median())
    _, edges = pd.qcut(spend, q=4, retbins=True, duplicates="drop")
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    return FeatureMetadata(
        spend_bin_edges=edges.tolist(),
        spend_bin_labels=["Low", "Mid-Low", "Mid-High", "High"],
        tenure_bin_edges=[-np.inf, 6, 12, 24, 48, np.inf],
        tenure_bin_labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    )


def validate_raw_features(df: pd.DataFrame) -> None:
    missing = [column for column in RAW_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def add_deployment_features(df: pd.DataFrame, metadata: FeatureMetadata | dict) -> pd.DataFrame:
    validate_raw_features(df)
    meta = metadata if isinstance(metadata, FeatureMetadata) else FeatureMetadata(**{
        "spend_bin_edges": metadata["spend_bin_edges"],
        "spend_bin_labels": metadata["spend_bin_labels"],
        "tenure_bin_edges": metadata["tenure_bin_edges"],
        "tenure_bin_labels": metadata["tenure_bin_labels"],
    })
    result = df[RAW_FEATURES].copy()
    result["tenure_cohort"] = pd.cut(
        pd.to_numeric(result["tenure_months"], errors="coerce"),
        bins=meta.tenure_bin_edges,
        labels=meta.tenure_bin_labels,
        include_lowest=True,
    ).astype("object")
    spend = pd.to_numeric(result["monthly_spend"], errors="coerce")
    result["spend_band"] = pd.cut(
        spend,
        bins=meta.spend_bin_edges,
        labels=meta.spend_bin_labels,
        include_lowest=True,
    ).astype("object")
    return result[MODEL_FEATURES]
