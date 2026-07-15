from pathlib import Path

from src.constants import DEFAULT_RECORD
from src.prediction_pipeline import DynamicPricingPipeline


def test_saved_model_generates_business_recommendation():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    pipeline = DynamicPricingPipeline(model_dir)
    result = pipeline.recommend_one(DEFAULT_RECORD, objective_label="Maximize Margin", n_grid=9)
    assert result["recommended_price"] >= DEFAULT_RECORD["base_cost"] * 1.05
    assert result["predicted_demand_optimized"] >= 0
    assert result["pricing_action"] in {"Maintain Price", "Increase Price", "Decrease Price", "Promotional Discount", "Manual Review"}
    assert not result["scenario_table"].empty
