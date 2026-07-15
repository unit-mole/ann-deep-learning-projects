from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentThresholds:
    low_max: float
    medium_max: float
    high_max: float

    def __post_init__(self) -> None:
        if not (0 <= self.low_max <= self.medium_max <= self.high_max):
            raise ValueError("Segment thresholds must be non-negative and ordered.")

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> "SegmentThresholds":
        return cls(
            low_max=float(values["low_max"]),
            medium_max=float(values["medium_max"]),
            high_max=float(values["high_max"]),
        )


SEGMENT_ORDER = [
    "Low Value Customer",
    "Medium Value Customer",
    "High Value Customer",
    "VIP / Strategic Customer",
]


def assign_clv_segment(value: float, thresholds: SegmentThresholds) -> str:
    value = max(float(value), 0.0)
    if value <= thresholds.low_max:
        return SEGMENT_ORDER[0]
    if value <= thresholds.medium_max:
        return SEGMENT_ORDER[1]
    if value <= thresholds.high_max:
        return SEGMENT_ORDER[2]
    return SEGMENT_ORDER[3]


def assign_clv_segments(values: Iterable[float], thresholds: SegmentThresholds) -> pd.Series:
    array = np.maximum(np.asarray(list(values), dtype=float), 0.0)
    labels = np.select(
        [
            array <= thresholds.low_max,
            array <= thresholds.medium_max,
            array <= thresholds.high_max,
        ],
        SEGMENT_ORDER[:3],
        default=SEGMENT_ORDER[3],
    )
    return pd.Series(labels, dtype="object")


def business_recommendation(segment: str, retention_probability: float) -> str:
    probability = float(np.clip(retention_probability, 0.0, 1.0))
    at_risk = probability < 0.50

    if segment == "VIP / Strategic Customer":
        return (
            "Launch an immediate high-touch retention plan and executive outreach."
            if at_risk
            else "Protect with VIP service, loyalty benefits, and premium cross-sell offers."
        )
    if segment == "High Value Customer":
        return (
            "Prioritize a targeted retention campaign before pursuing upsell."
            if at_risk
            else "Use personalized retention, premium offers, and loyalty enrollment."
        )
    if segment == "Medium Value Customer":
        return (
            "Run a reactivation journey with a relevant incentive."
            if at_risk
            else "Nurture with cross-sell offers and programs that increase purchase frequency."
        )
    return (
        "Use a low-cost automated win-back campaign."
        if at_risk
        else "Use efficient lifecycle messaging and monitor for growth signals."
    )
