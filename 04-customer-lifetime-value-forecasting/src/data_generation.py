from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticDataConfig:
    n_customers: int = 6000
    min_orders: int = 2
    max_orders: int = 25
    seed: int = 42
    reference_date: str = "2025-01-01"


def generate_synthetic_transactions(config: SyntheticDataConfig = SyntheticDataConfig()) -> pd.DataFrame:
    """Generate privacy-safe transactions with customer, order, and engagement fields."""
    rng = np.random.default_rng(config.seed)
    countries = ["United Kingdom", "Germany", "France", "Spain", "Netherlands", "Belgium", "Ireland"]
    channels = ["Web", "Mobile App", "Email Campaign", "Marketplace"]
    categories = ["Home", "Gift", "Electronics", "Beauty", "Office", "Fashion", "Kids"]
    payment_types = ["Card", "PayPal", "Wallet", "Bank Transfer"]
    reference_date = pd.Timestamp(config.reference_date)

    master = pd.DataFrame({
        "customer_id": [f"CUST_{i:05d}" for i in range(1, config.n_customers + 1)],
        "country": rng.choice(countries, config.n_customers, p=[0.40, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08]),
        "preferred_channel": rng.choice(channels, config.n_customers, p=[0.45, 0.30, 0.15, 0.10]),
        "loyalty_score": np.clip(rng.normal(60, 18, config.n_customers), 5, 100),
        "discount_sensitivity": np.clip(rng.beta(2, 5, config.n_customers), 0.01, 0.99),
        "engagement_score": np.clip(rng.normal(55, 20, config.n_customers), 1, 100),
    })

    category_price = {"Home": 18, "Gift": 12, "Electronics": 45, "Beauty": 22, "Office": 16, "Fashion": 30, "Kids": 20}
    records: list[dict] = []
    for customer in master.itertuples(index=False):
        n_orders = int(rng.integers(config.min_orders, config.max_orders + 1))
        customer_effect = 0.018 * customer.loyalty_score + 0.015 * customer.engagement_score - 8.0 * customer.discount_sensitivity
        for _ in range(n_orders):
            category = str(rng.choice(categories))
            quantity = int(rng.integers(1, 8))
            unit_price = max(2.0, float(rng.normal(category_price[category] + customer_effect, 6.0)))
            discount_rate = float(rng.beta(2, 6) * customer.discount_sensitivity)
            order_value = quantity * unit_price * (1 - discount_rate)
            records.append({
                "customer_id": customer.customer_id,
                "country": customer.country,
                "preferred_channel": customer.preferred_channel,
                "transaction_channel": str(rng.choice(channels, p=[0.42, 0.31, 0.15, 0.12])),
                "payment_type": str(rng.choice(payment_types)),
                "product_category": category,
                "order_date": reference_date - pd.Timedelta(days=int(rng.integers(1, 730))),
                "quantity": quantity,
                "unit_price": round(unit_price, 2),
                "discount_rate": round(discount_rate, 4),
                "order_value": round(order_value, 2),
                "loyalty_score": round(float(customer.loyalty_score), 2),
                "discount_sensitivity": round(float(customer.discount_sensitivity), 4),
                "engagement_score": round(float(customer.engagement_score), 2),
            })
    return pd.DataFrame(records).sort_values(["customer_id", "order_date"]).reset_index(drop=True)
