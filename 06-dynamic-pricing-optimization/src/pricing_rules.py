"""Transparent guardrails and action labels for price recommendations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceBounds:
    minimum: float
    maximum: float
    notes: tuple[str, ...]


def calculate_price_bounds(
    *,
    current_price: float,
    base_cost: float,
    competitor_price: float,
    inventory_level: float,
    baseline_demand: float,
    max_increase_pct: float = 0.20,
    max_decrease_pct: float = 0.30,
    minimum_markup_pct: float = 0.05,
) -> PriceBounds:
    """Create candidate bounds that implement explainable pricing guardrails."""

    notes: list[str] = []
    cost_floor = base_cost * (1 + minimum_markup_pct)
    minimum = max(cost_floor, current_price * (1 - max_decrease_pct))
    maximum = max(minimum, current_price * (1 + max_increase_pct))

    if inventory_level < max(baseline_demand * 0.75, 1.0):
        minimum = max(minimum, current_price * 0.95)
        notes.append("Low inventory restricted aggressive discounting.")

    if competitor_price < current_price * 0.82:
        maximum = min(maximum, max(current_price, competitor_price * 1.18))
        maximum = max(maximum, minimum)
        notes.append("A large competitor price gap limited the upward price range.")

    if minimum == cost_floor:
        notes.append("The recommendation cannot fall below cost plus a 5% minimum markup.")

    return PriceBounds(float(minimum), float(maximum), tuple(notes))


def determine_pricing_action(
    *,
    current_price: float,
    recommended_price: float,
    inventory_level: float,
    baseline_demand: float,
    competitor_price: float,
) -> str:
    change_pct = (recommended_price - current_price) / max(current_price, 1e-6)

    if recommended_price < 0 or baseline_demand < 0:
        return "Manual Review"
    if recommended_price > competitor_price * 1.20 and baseline_demand < 20:
        return "Manual Review"
    if abs(change_pct) <= 0.03:
        return "Maintain Price"
    if change_pct >= 0.07:
        return "Increase Price"
    if change_pct <= -0.12 and inventory_level > baseline_demand * 2.0:
        return "Promotional Discount"
    if change_pct < -0.03:
        return "Decrease Price"
    return "Maintain Price"


def build_business_recommendation(
    *,
    action: str,
    change_pct: float,
    inventory_level: float,
    optimized_demand: float,
    competitor_price: float,
    recommended_price: float,
    objective_label: str,
    rule_notes: tuple[str, ...] = (),
) -> str:
    if action == "Increase Price":
        message = (
            f"Increase price by approximately {change_pct:.1%}. The model expects sufficient demand to support a higher "
            f"price under the '{objective_label}' objective."
        )
    elif action == "Promotional Discount":
        message = (
            f"Use a controlled promotional discount of approximately {abs(change_pct):.1%}. Inventory is high relative "
            "to expected demand, so a temporary price reduction may improve sell-through."
        )
    elif action == "Decrease Price":
        message = (
            f"Decrease price by approximately {abs(change_pct):.1%} to improve demand efficiency while respecting the "
            "cost floor and configured change limits."
        )
    elif action == "Manual Review":
        message = (
            "Escalate this recommendation for manual review because the competitive position or predicted demand creates "
            "an unusually high-risk pricing scenario."
        )
    else:
        message = "Maintain the current price because nearby alternatives do not create a material improvement in the selected objective."

    if recommended_price > competitor_price * 1.10:
        message += " The recommended price remains more than 10% above the competitor reference, so monitor conversion closely."
    if inventory_level < optimized_demand:
        message += " Inventory may constrain realized sales; confirm replenishment capacity before publishing the price."
    if rule_notes:
        message += " " + " ".join(rule_notes)
    return message
