from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_synthetic_tabular_data(
    n_samples: int = 25_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the deterministic mixed-type dataset used by the project.

    The target is a synthetic positive-outcome propensity. The generator is
    intentionally transparent so the repository can be reproduced without
    publishing private or licensed business data.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "region": rng.choice(
                ["North", "South", "East", "West", "Central", "Coastal"],
                size=n_samples,
                p=[0.18, 0.16, 0.15, 0.17, 0.14, 0.20],
            ),
            "education_level": rng.choice(
                ["HighSchool", "Diploma", "Bachelors", "Masters", "PhD"],
                size=n_samples,
                p=[0.28, 0.16, 0.30, 0.20, 0.06],
            ),
            "occupation": rng.choice(
                [
                    "Sales", "Engineering", "Operations", "Healthcare",
                    "Finance", "Education", "Marketing", "HR", "Support",
                    "Legal", "Retail",
                ],
                size=n_samples,
            ),
            "channel": rng.choice(
                ["Web", "Mobile", "Branch", "Partner"],
                size=n_samples,
                p=[0.42, 0.38, 0.10, 0.10],
            ),
            "device_type": rng.choice(
                ["Desktop", "Mobile", "Tablet"],
                size=n_samples,
                p=[0.35, 0.55, 0.10],
            ),
            "membership_tier": rng.choice(
                ["Bronze", "Silver", "Gold", "Platinum"],
                size=n_samples,
                p=[0.46, 0.28, 0.18, 0.08],
            ),
            "product_category": rng.choice(list("ABCDEFG"), size=n_samples),
            "age": np.clip(rng.normal(38, 11, size=n_samples), 18, 75).round().astype(int),
            "years_experience": np.clip(rng.normal(9, 6, size=n_samples), 0, 40),
            "monthly_sessions": np.clip(rng.normal(18, 8, size=n_samples), 1, 60),
            "avg_basket_value": np.clip(rng.normal(220, 95, size=n_samples), 10, 1_200),
            "days_since_last_activity": np.clip(rng.exponential(25, size=n_samples), 0, 180),
            "credit_utilization": np.clip(rng.beta(2.5, 3.5, size=n_samples), 0, 1),
            "income_estimate": np.clip(rng.normal(62_000, 22_000, size=n_samples), 12_000, 180_000),
        }
    )

    region_effect = {
        "North": 0.15, "South": -0.10, "East": 0.02,
        "West": 0.12, "Central": -0.04, "Coastal": 0.18,
    }
    education_effect = {
        "HighSchool": -0.35, "Diploma": -0.15, "Bachelors": 0.18,
        "Masters": 0.32, "PhD": 0.42,
    }
    tier_effect = {"Bronze": -0.28, "Silver": 0.02, "Gold": 0.24, "Platinum": 0.48}
    channel_effect = {"Web": 0.08, "Mobile": 0.04, "Branch": -0.03, "Partner": 0.10}

    linear_score = (
        -3.7
        + 0.000018 * df["income_estimate"]
        + 0.0045 * df["avg_basket_value"]
        + 0.035 * df["monthly_sessions"]
        - 0.020 * df["days_since_last_activity"]
        - 0.90 * df["credit_utilization"]
        + 0.055 * df["years_experience"]
        + 0.012 * df["age"]
        + df["region"].map(region_effect)
        + df["education_level"].map(education_effect)
        + df["membership_tier"].map(tier_effect)
        + df["channel"].map(channel_effect)
    )

    interaction_bonus = (
        (
            df["education_level"].isin(["Masters", "PhD"])
            & df["membership_tier"].isin(["Gold", "Platinum"])
        ).astype(int) * 0.42
        + ((df["device_type"] == "Mobile") & (df["channel"] == "Mobile")).astype(int) * 0.18
        + (
            df["occupation"].isin(["Engineering", "Finance", "Legal"])
            & (df["income_estimate"] > 85_000)
        ).astype(int) * 0.28
        - (
            (df["days_since_last_activity"] > 60)
            & (df["monthly_sessions"] < 8)
        ).astype(int) * 0.55
    )

    logits = linear_score + interaction_bonus + rng.normal(0, 0.55, size=n_samples)
    df["target"] = rng.binomial(1, _sigmoid(logits))
    return df
