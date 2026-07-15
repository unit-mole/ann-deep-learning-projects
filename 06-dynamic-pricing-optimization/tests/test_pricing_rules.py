from src.pricing_rules import calculate_price_bounds, determine_pricing_action


def test_price_bounds_respect_cost_floor_and_change_limits():
    bounds = calculate_price_bounds(
        current_price=100.0,
        base_cost=90.0,
        competitor_price=105.0,
        inventory_level=500.0,
        baseline_demand=100.0,
    )
    assert bounds.minimum >= 94.5
    assert bounds.minimum >= 70.0
    assert bounds.maximum <= 120.0


def test_low_inventory_prevents_large_discount():
    bounds = calculate_price_bounds(
        current_price=100.0,
        base_cost=40.0,
        competitor_price=100.0,
        inventory_level=20.0,
        baseline_demand=100.0,
    )
    assert bounds.minimum >= 95.0


def test_action_labels():
    assert determine_pricing_action(current_price=100, recommended_price=112, inventory_level=200, baseline_demand=50, competitor_price=115) == "Increase Price"
    assert determine_pricing_action(current_price=100, recommended_price=101, inventory_level=200, baseline_demand=50, competitor_price=100) == "Maintain Price"
