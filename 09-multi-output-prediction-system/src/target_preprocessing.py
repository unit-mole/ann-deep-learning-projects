from __future__ import annotations

import numpy as np


def fit_regression_scalers(clv_train: np.ndarray, engagement_train: np.ndarray) -> dict[str, float]:
    return {
        "clv_mean": float(np.mean(clv_train)),
        "clv_std": float(np.std(clv_train)),
        "engagement_mean": float(np.mean(engagement_train)),
        "engagement_std": float(np.std(engagement_train)),
    }


def scale_regression_targets(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / std


def inverse_scale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean
