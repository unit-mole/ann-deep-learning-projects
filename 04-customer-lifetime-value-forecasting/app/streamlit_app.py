from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, OUTPUTS_DIR
from src.feature_engineering import add_derived_features, infer_legacy_customer_segment
from src.prediction_pipeline import CLVPredictionPipeline, PredictionError

st.set_page_config(
    page_title="Customer Lifetime Value Forecasting",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: .7rem; border-radius: .5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline() -> CLVPredictionPipeline:
    return CLVPredictionPipeline()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_input.csv")


@st.cache_data
def load_metrics() -> dict:
    return json.loads((OUTPUTS_DIR / "model_metrics.json").read_text(encoding="utf-8"))


try:
    pipeline = load_pipeline()
except PredictionError as exc:
    st.error(str(exc))
    st.stop()

st.title("Customer Lifetime Value Forecasting")
st.caption(
    "Multi-task ANN prototype that forecasts next-90-day customer value and retention probability, "
    "then translates the outputs into business-friendly segments and actions."
)

with st.sidebar:
    st.header("Model scope")
    st.write("**Forecast horizon:** 90 days")
    st.write("**Training data:** privacy-safe synthetic transactions")
    st.write("**Outputs:** customer value, retention probability, segment, recommendation")
    st.warning("This is a portfolio prototype and not a production financial forecast.")

overview_tab, single_tab, batch_tab, performance_tab = st.tabs(
    ["Overview", "Single Customer", "Batch Scoring", "Model Performance"]
)

with overview_tab:
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Business question")
        st.markdown(
            "> Given a customer's profile and observed purchase behavior, what revenue might the "
            "customer generate in the next 90 days, and what action should the business take?"
        )
        st.subheader("Workflow")
        st.markdown(
            """
1. Aggregate customer behavior into RFM, value, diversity, cohort, and engagement features.
2. Scale numerical features and embed categorical variables.
3. Predict 90-day customer value and retention probability with a shared ANN.
4. Convert the value forecast into Low, Medium, High, or VIP segments.
5. Attach a retention or growth recommendation.
            """
        )
    with right:
        metrics = load_metrics()
        st.subheader("Supplied-model test results")
        c1, c2 = st.columns(2)
        c1.metric("MAE", f"${metrics['regression']['mae']:,.2f}")
        c2.metric("RMSE", f"${metrics['regression']['rmse']:,.2f}")
        c1.metric("R²", f"{metrics['regression']['r2']:.3f}")
        c2.metric("Retention ROC-AUC", f"{metrics['retention']['roc_auc']:.3f}")
        st.info(
            "The R² indicates a baseline with modest predictive fit. The project emphasizes honest "
            "evaluation, deployment, and a clear path to improvement rather than overstating accuracy."
        )

with single_tab:
    st.subheader("Score one customer")
    defaults = pipeline.numeric_defaults
    options = pipeline.category_options
    with st.form("single_customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            customer_id = st.text_input("Customer ID", "DEMO_SINGLE")
            n_orders = st.number_input("Number of orders", min_value=1, value=12, step=1)
            total_revenue = st.number_input("Historical total revenue", min_value=0.0, value=950.0, step=25.0)
            std_order_value = st.number_input("Order value standard deviation", min_value=0.0, value=24.0, step=1.0)
            total_quantity = st.number_input("Total quantity", min_value=1, value=42, step=1)
            avg_discount = st.number_input("Average discount rate", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
        with c2:
            recency_days = st.number_input("Days since last purchase", min_value=0, value=18, step=1)
            tenure_days = st.number_input("Customer tenure (days)", min_value=1, value=310, step=10)
            category_diversity = st.number_input("Product category diversity", min_value=1, max_value=20, value=4, step=1)
            payment_diversity = st.number_input("Payment method diversity", min_value=1, max_value=10, value=2, step=1)
            channel_diversity = st.number_input("Channel diversity", min_value=1, max_value=10, value=3, step=1)
            loyalty_score = st.number_input("Loyalty score", min_value=0.0, max_value=100.0, value=74.0, step=1.0)
        with c3:
            engagement_score = st.number_input("Engagement score", min_value=0.0, max_value=100.0, value=81.0, step=1.0)
            discount_sensitivity = st.number_input("Discount sensitivity", min_value=0.0, max_value=1.0, value=0.22, step=0.01)
            country = st.selectbox("Country", options["country"], index=options["country"].index("United Kingdom"))
            preferred_channel = st.selectbox("Preferred channel", options["preferred_channel"], index=options["preferred_channel"].index("Web"))
            dominant_category = st.selectbox("Dominant category", options["dominant_category"], index=options["dominant_category"].index("Electronics"))
            acquisition_quarter = st.selectbox("Acquisition quarter", options["acquisition_quarter"], index=options["acquisition_quarter"].index("2024Q1"))

        override_segment = st.checkbox("Override automatically inferred legacy segment")
        manual_segment = st.selectbox("Legacy model segment", options["customer_segment_name"], disabled=not override_segment)
        submitted = st.form_submit_button("Forecast customer value", type="primary")

    if submitted:
        avg_order_value = total_revenue / max(n_orders, 1)
        avg_quantity = total_quantity / max(n_orders, 1)
        tenure_months = max(tenure_days / 30.0, 1.0)
        record = {
            "customer_id": customer_id,
            "n_orders": n_orders,
            "total_revenue": total_revenue,
            "avg_order_value": avg_order_value,
            "std_order_value": std_order_value,
            "total_quantity": total_quantity,
            "avg_quantity": avg_quantity,
            "category_diversity": category_diversity,
            "payment_diversity": payment_diversity,
            "channel_diversity": channel_diversity,
            "avg_discount": avg_discount,
            "recency_days": recency_days,
            "tenure_days": tenure_days,
            "orders_per_month": n_orders / tenure_months,
            "revenue_per_month": total_revenue / tenure_months,
            "loyalty_score": loyalty_score,
            "discount_sensitivity": discount_sensitivity,
            "engagement_score": engagement_score,
            "cohort_age_months": tenure_days / 30.44,
            "avg_revenue_per_active_month": total_revenue / max(tenure_days / 30.44, 1),
            "country": country,
            "preferred_channel": preferred_channel,
            "dominant_category": dominant_category,
            "acquisition_quarter": acquisition_quarter,
        }
        inferred = infer_legacy_customer_segment(pd.DataFrame([record])).iloc[0]
        record["customer_segment_name"] = manual_segment if override_segment else inferred
        try:
            result = pipeline.predict_single(record)
            a, b, c = st.columns(3)
            a.metric("Predicted 90-day CLV", f"${result['predicted_clv_90d']:,.2f}")
            b.metric("Customer segment", result["customer_value_segment"])
            c.metric("Retention probability", f"{result['predicted_retention_probability']:.1%}")
            st.success(result["business_recommendation"])
            with st.expander("Scored customer details"):
                st.json(result)
        except PredictionError as exc:
            st.error(str(exc))

with batch_tab:
    st.subheader("Batch customer scoring")
    source = st.radio("Input source", ["Use included sample", "Upload CSV"], horizontal=True)
    if source == "Use included sample":
        batch_input = load_sample()
    else:
        upload = st.file_uploader("Upload a CSV with customer-level features", type=["csv"])
        batch_input = pd.read_csv(upload) if upload is not None else None

    if batch_input is not None:
        st.write("Input preview")
        st.dataframe(batch_input.head(20), use_container_width=True)
        if len(batch_input) > 10000:
            st.error("For this public demo, upload at most 10,000 customer rows at a time.")
        elif st.button("Run batch forecast", type="primary"):
            try:
                scored = pipeline.predict_dataframe(batch_input)
                if pipeline.last_warnings:
                    with st.expander(f"Data preparation notes ({len(pipeline.last_warnings)})"):
                        for note in pipeline.last_warnings:
                            st.write(f"- {note}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Customers scored", f"{len(scored):,}")
                m2.metric("Average predicted CLV", f"${scored['predicted_clv_90d'].mean():,.2f}")
                m3.metric("VIP customers", int((scored['customer_value_segment'] == 'VIP / Strategic Customer').sum()))
                m4.metric("At-risk customers", int((scored['predicted_retention_probability'] < 0.50).sum()))

                left, right = st.columns(2)
                with left:
                    st.write("Segment distribution")
                    segment_counts = scored["customer_value_segment"].value_counts()
                    st.bar_chart(segment_counts)
                with right:
                    st.write("Predicted CLV distribution")
                    bins = np.histogram_bin_edges(scored["predicted_clv_90d"], bins=min(20, max(5, int(np.sqrt(len(scored))))))
                    labels = [f"${bins[i]:.0f}–${bins[i+1]:.0f}" for i in range(len(bins) - 1)]
                    counts, _ = np.histogram(scored["predicted_clv_90d"], bins=bins)
                    st.bar_chart(pd.Series(counts, index=labels))

                st.dataframe(
                    scored.sort_values("predicted_clv_90d", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download scored CSV",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="clv_scored_customers.csv",
                    mime="text/csv",
                )
            except (PredictionError, ValueError) as exc:
                st.error(str(exc))

with performance_tab:
    metrics = load_metrics()
    st.subheader("Evaluation summary")
    st.write(
        "Metrics below are recalculated from the supplied held-out prediction artifact. "
        "MAPE is reported only for non-zero actual values because 24.9% of the test customers had zero future revenue."
    )
    regression = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²", "Non-zero MAPE", "sMAPE", "WAPE"],
        "Value": [
            f"${metrics['regression']['mae']:,.2f}",
            f"${metrics['regression']['rmse']:,.2f}",
            f"{metrics['regression']['r2']:.4f}",
            f"{metrics['regression']['mape_nonzero_actuals_pct']:.1f}%",
            f"{metrics['regression']['smape_pct']:.1f}%",
            f"{metrics['regression']['wape_pct']:.1f}%",
        ],
    })
    retention = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Value": [f"{metrics['retention'][key]:.4f}" for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]],
    })
    a, b = st.columns(2)
    a.dataframe(regression, hide_index=True, use_container_width=True)
    b.dataframe(retention, hide_index=True, use_container_width=True)

    for image_name, caption in [
        ("actual_vs_predicted.png", "Actual vs predicted customer value"),
        ("residual_plot.png", "Residual analysis"),
        ("clv_distribution.png", "Actual and predicted distributions"),
        ("customer_segment_distribution.png", "Value-segment distribution"),
        ("training_history.png", "Training and validation MAE"),
        ("retention_confusion_matrix.png", "Retention confusion matrix"),
    ]:
        image_path = OUTPUTS_DIR / image_name
        if image_path.exists():
            st.image(str(image_path), caption=caption, use_container_width=True)

    st.warning(
        "The model underpredicts the highest-value customers and explains limited revenue variance. "
        "Recommended next steps are a two-stage zero-inflated design, log-target retraining, temporal validation, "
        "and comparison against gradient-boosted tree baselines."
    )
