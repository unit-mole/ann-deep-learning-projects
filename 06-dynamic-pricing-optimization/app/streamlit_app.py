"""Interactive Streamlit application for ANN-based dynamic pricing optimization."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CATEGORIES, CHANNELS, DEFAULT_RECORD, OBJECTIVES, REGIONS
from src.data_preprocessing import validate_input_columns
from src.prediction_pipeline import DynamicPricingPipeline

MODEL_DIR = PROJECT_ROOT / "models"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_input.csv"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

st.set_page_config(
    page_title="Dynamic Pricing Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading trained ANN models...")
def load_pipeline() -> DynamicPricingPipeline:
    return DynamicPricingPipeline(MODEL_DIR)


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data
def load_model_metrics() -> dict:
    """Load evaluation metrics from the canonical saved-model artifact."""
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    required_regression_metrics = {"mae", "rmse", "r2", "mape_pct"}
    regression_metrics = metrics.get("regression", {})
    missing_metrics = required_regression_metrics.difference(regression_metrics)
    if missing_metrics:
        missing = ", ".join(sorted(missing_metrics))
        raise ValueError(f"Missing regression metrics in {METRICS_PATH.name}: {missing}")

    return metrics


def money(value: float) -> str:
    return f"${value:,.2f}"


def build_scenario_chart(scenarios: pd.DataFrame, recommended_price: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scenarios["current_price"],
            y=scenarios["predicted_demand"],
            mode="lines+markers",
            name="Predicted demand",
            yaxis="y",
            hovertemplate="Price: $%{x:,.2f}<br>Demand: %{y:,.1f} units<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scenarios["current_price"],
            y=scenarios["predicted_revenue"],
            mode="lines",
            name="Predicted revenue",
            yaxis="y2",
            hovertemplate="Price: $%{x:,.2f}<br>Revenue: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_vline(x=recommended_price, line_dash="dash", annotation_text="Recommended")
    fig.update_layout(
        title="Price sensitivity and revenue curve",
        xaxis_title="Candidate price",
        yaxis=dict(title="Predicted demand (units)"),
        yaxis2=dict(title="Predicted revenue", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        height=480,
    )
    return fig


def show_recommendation(result: dict) -> None:
    st.subheader("Pricing recommendation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current price", money(result["current_price"]))
    c2.metric(
        "Recommended price",
        money(result["recommended_price"]),
        f'{result["price_change_pct"]:+.1%}',
    )
    c3.metric(
        "Optimized demand",
        f'{result["predicted_demand_optimized"]:,.1f} units',
        f'{result["predicted_demand_optimized"] - result["predicted_demand_current"]:+,.1f}',
    )
    c4.metric(
        "Estimated margin",
        money(result["predicted_margin_optimized"]),
        money(result["estimated_margin_uplift"]),
    )

    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"**Pricing action:** `{result['pricing_action']}`")
        st.markdown(f"**Pricing segment:** {result['pricing_segment']}")
        st.markdown(f"**High-demand probability:** {result['high_demand_probability']:.1%}")
        st.markdown(f"**Optimized revenue:** {money(result['predicted_revenue_optimized'])}")
        st.info(result["business_recommendation"])
        if result["rule_notes"]:
            st.caption("Guardrails applied: " + " ".join(result["rule_notes"]))
    with right:
        st.plotly_chart(
            build_scenario_chart(result["scenario_table"], result["recommended_price"]),
            use_container_width=True,
        )

    scenario_display = result["scenario_table"].copy()
    scenario_display = scenario_display[
        [
            "current_price",
            "predicted_demand",
            "predicted_revenue",
            "predicted_margin",
            "price_change_pct",
            "selected",
        ]
    ].rename(
        columns={
            "current_price": "candidate_price",
            "selected": "recommended",
        }
    )
    st.dataframe(
        scenario_display.style.format(
            {
                "candidate_price": "${:,.2f}",
                "predicted_demand": "{:,.1f}",
                "predicted_revenue": "${:,.0f}",
                "predicted_margin": "${:,.0f}",
                "price_change_pct": "{:+.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def manual_record() -> dict:
    st.markdown("Enter decision-time product and market information. No future demand or optimal-price field is required.")
    a, b, c = st.columns(3)
    with a:
        product_id = st.text_input("Product ID", DEFAULT_RECORD["product_id"])
        category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(DEFAULT_RECORD["category"]))
        region = st.selectbox("Region", REGIONS, index=REGIONS.index(DEFAULT_RECORD["region"]))
        channel = st.selectbox("Channel", CHANNELS, index=CHANNELS.index(DEFAULT_RECORD["channel"]))
        day_of_year = st.number_input("Day of year", 1, 365, int(DEFAULT_RECORD["day_of_year"]))
        weekend_flag = int(st.checkbox("Weekend", bool(DEFAULT_RECORD["weekend_flag"])))
    with b:
        base_cost = st.number_input("Unit cost ($)", 0.01, 100000.0, float(DEFAULT_RECORD["base_cost"]), step=1.0)
        current_price = st.number_input("Current price ($)", 0.01, 100000.0, float(DEFAULT_RECORD["current_price"]), step=1.0)
        competitor_price = st.number_input("Competitor price ($)", 0.01, 100000.0, float(DEFAULT_RECORD["competitor_price"]), step=1.0)
        inventory_level = st.number_input("Inventory level", 0.0, 1_000_000.0, float(DEFAULT_RECORD["inventory_level"]), step=10.0)
        historical_sales = st.number_input("Historical sales", 0.0, 1_000_000.0, float(DEFAULT_RECORD["historical_sales"]), step=5.0)
        rating = st.slider("Customer rating", 1.0, 5.0, float(DEFAULT_RECORD["rating"]), 0.1)
    with c:
        marketing_index = st.slider("Marketing index", 0.0, 100.0, float(DEFAULT_RECORD["marketing_index"]), 1.0)
        demand_index = st.slider("Demand index", 0.0, 100.0, float(DEFAULT_RECORD["demand_index"]), 1.0)
        customer_income_index = st.slider("Customer income index", 1.0, 250.0, float(DEFAULT_RECORD["customer_income_index"]), 1.0)
        holiday_flag = int(st.checkbox("Holiday", bool(DEFAULT_RECORD["holiday_flag"])))
        promotion_flag = int(st.checkbox("Promotion active", bool(DEFAULT_RECORD["promotion_flag"])))
        competitor_promo_flag = int(st.checkbox("Competitor promotion active", bool(DEFAULT_RECORD["competitor_promo_flag"])))

    return {
        "product_id": product_id,
        "day_of_year": day_of_year,
        "weekend_flag": weekend_flag,
        "holiday_flag": holiday_flag,
        "category": category,
        "region": region,
        "channel": channel,
        "promotion_flag": promotion_flag,
        "competitor_promo_flag": competitor_promo_flag,
        "base_cost": base_cost,
        "current_price": current_price,
        "competitor_price": competitor_price,
        "rating": rating,
        "inventory_level": inventory_level,
        "marketing_index": marketing_index,
        "demand_index": demand_index,
        "customer_income_index": customer_income_index,
        "historical_sales": historical_sales,
    }


st.title("ANN Dynamic Pricing Optimization System")
st.caption(
    "Leakage-safe demand forecasting plus transparent price-scenario optimization, business guardrails, "
    "and downloadable recommendations."
)

with st.sidebar:
    st.header("Optimization settings")
    objective = st.selectbox("Business objective", list(OBJECTIVES), index=0)
    scenario_count = st.slider("Candidate price scenarios", 9, 41, 21, step=2)
    st.divider()
    st.markdown("**Model:** Keras 3 tabular ANN")
    st.markdown("**Inference backend:** PyTorch")
    st.markdown("**Data:** Reproducible synthetic retail data")
    st.warning("Portfolio demonstration only. Validate with causal experiments and production data before real pricing decisions.")

try:
    pipeline = load_pipeline()
except Exception as exc:
    st.error(f"The model artifacts could not be loaded: {exc}")
    st.stop()

tab_manual, tab_batch, tab_about = st.tabs(["Single product", "Batch optimization", "Model and methodology"])

with tab_manual:
    with st.form("manual_pricing_form"):
        record = manual_record()
        submitted = st.form_submit_button("Generate recommendation", type="primary", use_container_width=True)
    if submitted:
        try:
            with st.spinner("Scoring candidate prices..."):
                recommendation = pipeline.recommend_one(record, objective_label=objective, n_grid=scenario_count)
            show_recommendation(recommendation)
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")

with tab_batch:
    st.subheader("Optimize a product portfolio")
    source = st.radio("Data source", ["Use sample data", "Upload CSV"], horizontal=True)
    if source == "Upload CSV":
        upload = st.file_uploader("Upload pricing input", type=["csv"])
        batch_data = pd.read_csv(upload) if upload is not None else None
    else:
        batch_data = load_sample_data()

    if batch_data is not None:
        st.markdown("**Input preview**")
        st.dataframe(batch_data.head(25), use_container_width=True, hide_index=True)
        missing = validate_input_columns(batch_data)
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif len(batch_data) > 500:
            st.error("The hosted demo accepts up to 500 rows per run. Split larger files into smaller batches.")
        elif st.button("Optimize batch", type="primary", use_container_width=True):
            try:
                with st.spinner("Running vectorized price-scenario inference..."):
                    batch_result = pipeline.recommend_batch(
                        batch_data,
                        objective_label=objective,
                        n_grid=min(scenario_count, 25),
                    )
                a, b, c, d = st.columns(4)
                a.metric("Products optimized", f"{len(batch_result):,}")
                b.metric("Median price change", f'{batch_result["price_change_pct"].median():+.1%}')
                c.metric("Estimated margin uplift", money(batch_result["estimated_margin_uplift"].sum()))
                d.metric("Manual-review flags", int((batch_result["pricing_action"] == "Manual Review").sum()))

                action_summary = (
                    batch_result["pricing_action"].value_counts().rename_axis("pricing_action").reset_index(name="products")
                )
                st.bar_chart(action_summary.set_index("pricing_action"))
                st.dataframe(
                    batch_result.style.format(
                        {
                            "current_price": "${:,.2f}",
                            "recommended_price": "${:,.2f}",
                            "price_change_pct": "{:+.1%}",
                            "predicted_demand_current": "{:,.1f}",
                            "predicted_demand_optimized": "{:,.1f}",
                            "predicted_revenue_current": "${:,.0f}",
                            "predicted_revenue_optimized": "${:,.0f}",
                            "predicted_margin_current": "${:,.0f}",
                            "predicted_margin_optimized": "${:,.0f}",
                            "estimated_margin_uplift": "${:,.0f}",
                            "high_demand_probability": "{:.1%}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download optimized pricing CSV",
                    batch_result.to_csv(index=False).encode("utf-8"),
                    file_name="optimized_pricing_recommendations.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Batch optimization failed: {exc}")

with tab_about:
    st.subheader("Model architecture and validation")
    try:
        model_metrics = load_model_metrics()
        regression_metrics = model_metrics["regression"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Demand MAE", f'{float(regression_metrics["mae"]):.2f} units')
        m2.metric("Demand RMSE", f'{float(regression_metrics["rmse"]):.2f} units')
        m3.metric("Demand R²", f'{float(regression_metrics["r2"]):.3f}')
        m4.metric("Demand MAPE", f'{float(regression_metrics["mape_pct"]):.2f}%')
        st.caption(
            f'Metrics loaded from `models/model_metrics.json` and reported on '
            f'{int(model_metrics.get("test_rows", 0)):,} held-out rows.'
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        st.error(f"Model metrics could not be loaded: {exc}")

    st.markdown(
        """
        The production path predicts **demand at a proposed price**, rather than predicting an engineered optimal-price label.
        Candidate prices are generated within business guardrails and scored for expected demand, revenue, and gross margin.

        **Inputs:** calendar signals, category, region, channel, promotion flags, cost, proposed price, competitor price,
        rating, inventory, marketing index, demand index, customer-income index, and historical sales.

        **ANN:** a numerical branch plus categorical embedding branches, dense hidden layers, batch normalization,
        dropout, and a regression output trained with Huber loss on log-demand. A second ANN estimates high-demand probability.

        **Guardrails:** price above cost plus minimum markup; no more than a 20% increase or 30% decrease in one run;
        low-inventory discount protection; and competitor-gap review logic.
        """
    )
    st.image(str(PROJECT_ROOT / "outputs" / "actual_vs_predicted_demand.png"), caption="Held-out actual vs predicted demand")
    st.image(str(PROJECT_ROOT / "outputs" / "feature_importance.png"), caption="Permutation-based demand-driver analysis")
