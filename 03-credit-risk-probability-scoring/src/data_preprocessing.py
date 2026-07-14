from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, SEED,
    TARGET_COLUMN, TRUE_PROBABILITY_COLUMN,
)
from .feature_engineering import prepare_feature_frame


def generate_credit_risk_data(n_samples: int = 25_000, seed: int = SEED) -> pd.DataFrame:
    """Generate the deterministic synthetic banking dataset used in the original notebook."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n_samples)
    annual_income = np.clip(rng.normal(68_000, 26_000, n_samples), 12_000, 250_000)
    loan_amount = np.clip(rng.normal(18_000, 9_000, n_samples), 1_000, 70_000)
    interest_rate = np.clip(rng.normal(12.5, 5.0, n_samples), 4.5, 32.0)
    employment_length = rng.integers(0, 25, n_samples)
    credit_history_years = np.clip(rng.normal(8.0, 4.5, n_samples), 0.0, 35.0)
    revolving_utilization = np.clip(rng.beta(2.2, 4.0, n_samples) * 100, 0, 100)
    dti = np.clip(rng.normal(18, 9, n_samples), 0, 55)
    delinquency_count = rng.poisson(0.55, n_samples)
    inquiry_count = rng.poisson(1.8, n_samples)
    open_accounts = np.clip(rng.normal(8, 4, n_samples), 1, 30).astype(int)
    existing_loans = np.clip(rng.normal(2, 1.3, n_samples), 0, 8).astype(int)
    collateral_value = np.clip(rng.normal(12_000, 15_000, n_samples), 0, 120_000)

    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n_samples, p=[0.38, 0.40, 0.20, 0.02])
    purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "education", "small_business", "medical", "auto", "vacation"],
        n_samples, p=[0.35, 0.15, 0.10, 0.12, 0.08, 0.12, 0.08],
    )
    grade = rng.choice(["A", "B", "C", "D", "E", "F", "G"], n_samples, p=[0.10, 0.18, 0.24, 0.22, 0.14, 0.08, 0.04])
    region = rng.choice(["North", "South", "East", "West"], n_samples)
    verification_status = rng.choice(["verified", "not_verified", "source_verified"], n_samples, p=[0.36, 0.34, 0.30])

    monthly_income = annual_income / 12.0
    loan_to_income = loan_amount / np.maximum(annual_income, 1)
    collateral_ratio = collateral_value / np.maximum(loan_amount, 1)

    grade_risk = {"A": -1.0, "B": -0.5, "C": 0.0, "D": 0.45, "E": 0.95, "F": 1.35, "G": 1.75}
    purpose_risk = {"debt_consolidation": 0.20, "home_improvement": -0.05, "education": 0.15, "small_business": 0.80, "medical": 0.40, "auto": 0.05, "vacation": 0.35}
    home_risk = {"RENT": 0.25, "MORTGAGE": -0.10, "OWN": -0.20, "OTHER": 0.15}
    verification_risk = {"verified": -0.12, "not_verified": 0.18, "source_verified": -0.05}

    linear_score = (
        -4.2 + 3.8 * loan_to_income + 0.050 * interest_rate + 0.040 * dti
        + 0.020 * revolving_utilization + 0.22 * delinquency_count + 0.12 * inquiry_count
        - 0.030 * employment_length - 0.050 * credit_history_years - 0.000012 * annual_income
        - 0.18 * collateral_ratio + np.vectorize(grade_risk.get)(grade)
        + np.vectorize(purpose_risk.get)(purpose) + np.vectorize(home_risk.get)(home_ownership)
        + np.vectorize(verification_risk.get)(verification_status) + rng.normal(0, 0.55, n_samples)
    )
    probability_default = 1 / (1 + np.exp(-linear_score))
    default_flag = rng.binomial(1, probability_default, n_samples)

    return pd.DataFrame({
        "age": age, "annual_income": annual_income, "monthly_income": monthly_income,
        "loan_amount": loan_amount, "interest_rate": interest_rate,
        "employment_length": employment_length, "credit_history_years": credit_history_years,
        "revolving_utilization": revolving_utilization, "dti": dti,
        "delinquency_count": delinquency_count, "inquiry_count": inquiry_count,
        "open_accounts": open_accounts, "existing_loans": existing_loans,
        "collateral_value": collateral_value, "home_ownership": home_ownership,
        "purpose": purpose, "grade": grade, "region": region,
        "verification_status": verification_status, "loan_to_income": loan_to_income,
        "collateral_ratio": collateral_ratio, TRUE_PROBABILITY_COLUMN: probability_default,
        TARGET_COLUMN: default_flag,
    })


def inject_missing_values(data: pd.DataFrame, fraction: float = 0.01, seed: int = SEED) -> pd.DataFrame:
    result = data.copy()
    rng = np.random.default_rng(seed)
    for column in ["annual_income", "interest_rate", "dti", "employment_length", "home_ownership", "purpose"]:
        indices = rng.choice(result.index, size=max(50, int(len(result) * fraction)), replace=False)
        result.loc[indices, column] = np.nan
    return result


def load_dataset(csv_path: str | Path | None = None, n_samples: int = 25_000) -> pd.DataFrame:
    if csv_path is None:
        return inject_missing_values(generate_credit_risk_data(n_samples=n_samples))
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def split_dataset(data: pd.DataFrame):
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Training data must contain target column '{TARGET_COLUMN}'.")
    drop_columns = [column for column in [TRUE_PROBABILITY_COLUMN, TARGET_COLUMN] if column in data.columns]
    X = prepare_feature_frame(data.drop(columns=drop_columns))
    y = data[TARGET_COLUMN].astype(int)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1764705882,
        random_state=SEED, stratify=y_train_full,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def to_dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def fit_transform_splits(X_train, X_val, X_test):
    preprocessor = build_preprocessor()
    train = to_dense(preprocessor.fit_transform(X_train)).astype("float32")
    validation = to_dense(preprocessor.transform(X_val)).astype("float32")
    test = to_dense(preprocessor.transform(X_test)).astype("float32")
    return preprocessor, train, validation, test


def save_preprocessor(preprocessor: ColumnTransformer, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)


def export_portable_schema(preprocessor: ColumnTransformer, output_path: str | Path) -> dict[str, Any]:
    num_pipe = preprocessor.named_transformers_["num"]
    cat_pipe = preprocessor.named_transformers_["cat"]
    schema = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_medians": dict(zip(NUMERIC_FEATURES, map(float, num_pipe.named_steps["imputer"].statistics_))),
        "numeric_means": dict(zip(NUMERIC_FEATURES, map(float, num_pipe.named_steps["scaler"].mean_))),
        "numeric_scales": dict(zip(NUMERIC_FEATURES, map(float, num_pipe.named_steps["scaler"].scale_))),
        "categorical_modes": dict(zip(CATEGORICAL_FEATURES, map(str, cat_pipe.named_steps["imputer"].statistics_))),
        "categorical_categories": {
            feature: [str(value) for value in categories]
            for feature, categories in zip(CATEGORICAL_FEATURES, cat_pipe.named_steps["encoder"].categories_)
        },
        "processed_feature_names": preprocessor.get_feature_names_out().tolist(),
    }
    schema["processed_input_dim"] = len(schema["processed_feature_names"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema
