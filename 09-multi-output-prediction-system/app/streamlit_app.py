from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RAW_FEATURES
from src.prediction_pipeline import MultiOutputPredictionPipeline


@st.cache_resource
def load_pipeline():
    return MultiOutputPredictionPipeline(PROJECT_ROOT)


def manual_input() -> pd.DataFrame:
    with st.form("customer_form"):
        left, middle, right = st.columns(3)
        with left:
            age = st.number_input("Age", 18, 90, 35)
            tenure = st.number_input("Tenure (months)", 1, 180, 24)
            income = st.number_input("Monthly income", 0.0, 50000.0, 6500.0, 100.0)
            spend = st.number_input("Monthly spend", 0.0, 5000.0, 420.0, 10.0)
            transactions = st.number_input("Transactions, last 30 days", 0, 100, 8)
            sessions = st.number_input("Average session minutes", 0.0, 180.0, 24.0, 1.0)
        with middle:
            products = st.number_input("Products owned", 0, 20, 3)
            complaints = st.number_input("Complaints, last 6 months", 0, 20, 1)
            days_login = st.number_input("Days since last login", 0, 365, 14)
            discount = st.slider("Discount usage rate", 0.0, 1.0, 0.30, 0.01)
            tickets = st.number_input("Support tickets, last 90 days", 0, 30, 1)
            visits = st.number_input("Website visits, last 30 days", 0, 150, 14)
        with right:
            region = st.selectbox("Region", ["North","South","East","West","Central"], index=3)
            channel = st.selectbox("Channel", ["Web","Mobile","Store","Partner"], index=1)
            plan = st.selectbox("Plan type", ["Basic","Standard","Premium","Enterprise"], index=1)
            segment = st.selectbox("Customer segment", ["Budget","Growth","Loyal","HighValue"], index=1)
            device = st.selectbox("Device type", ["Android","iOS","Desktop","Tablet"], index=1)
        submitted = st.form_submit_button("Generate multi-output prediction", type="primary", use_container_width=True)
    if not submitted:
        return pd.DataFrame()
    return pd.DataFrame([{
        "age":age,"tenure_months":tenure,"monthly_income":income,"monthly_spend":spend,
        "transactions_last_30d":transactions,"avg_session_minutes":sessions,"products_owned":products,
        "complaints_last_6m":complaints,"days_since_last_login":days_login,"discount_usage_rate":discount,
        "support_tickets_last_90d":tickets,"website_visits_last_30d":visits,"region":region,
        "channel":channel,"plan_type":plan,"segment":segment,"device_type":device,
    }])


def display_single(result: pd.Series) -> None:
    c1,c2,c3 = st.columns(3)
    c1.metric("Churn probability", f"{result['predicted_churn_probability']:.1%}", result["churn_risk_bucket"] + " risk")
    c2.metric("Predicted CLV", f"${result['predicted_clv']:,.0f}", result["clv_bucket"])
    c3.metric("Engagement", f"{result['predicted_engagement']:.1f} / 100", result["engagement_bucket"])
    st.info(result["recommended_action"])
    with st.expander("Scored record"):
        st.dataframe(result.to_frame("value"), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Multi-Output ANN", page_icon="🧠", layout="wide")
    st.title("Multi-Output Customer Intelligence")
    st.caption("One shared ANN predicts churn probability, customer lifetime value, and engagement score.")
    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error("The model artifacts could not be loaded. Install the pinned requirements and verify the models folder.")
        st.exception(exc)
        st.stop()
    tab_manual, tab_batch, tab_performance = st.tabs(["Manual prediction","Batch scoring","Model performance"])
    with tab_manual:
        input_df = manual_input()
        if not input_df.empty:
            display_single(pipeline.predict(input_df).iloc[0])
    with tab_batch:
        st.write("Upload a CSV with the 17 required raw input columns, or use the included sample.")
        source = st.radio("Input source", ["Included sample","Upload CSV"], horizontal=True)
        if source == "Included sample":
            batch = pd.read_csv(PROJECT_ROOT/"data"/"sample_batch_input.csv")
        else:
            uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
            batch = pd.read_csv(uploaded) if uploaded is not None else pd.DataFrame()
        if not batch.empty:
            st.dataframe(batch.head(20), use_container_width=True)
            if st.button("Score batch", type="primary"):
                missing = [column for column in RAW_FEATURES if column not in batch.columns]
                if missing:
                    st.error("Missing columns: " + ", ".join(missing))
                else:
                    scored = pipeline.predict(batch)
                    a,b,c = st.columns(3)
                    a.metric("Customers scored", f"{len(scored):,}")
                    b.metric("High-risk share", f"{(scored['churn_risk_bucket']=='High').mean():.1%}")
                    c.metric("Average predicted CLV", f"${scored['predicted_clv'].mean():,.0f}")
                    st.dataframe(scored, use_container_width=True)
                    st.bar_chart(scored["churn_risk_bucket"].value_counts())
                    st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"), "multi_output_scored.csv", "text/csv")
    with tab_performance:
        metrics = json.loads((PROJECT_ROOT/"outputs"/"model_metrics.json").read_text())
        st.subheader("Held-out benchmark")
        m1,m2,m3 = st.columns(3)
        m1.metric("Churn ROC-AUC", f"{metrics['churn']['roc_auc']:.3f}")
        m2.metric("CLV R²", f"{metrics['clv']['r2']:.3f}")
        m3.metric("Engagement R²", f"{metrics['engagement']['r2']:.3f}")
        st.warning("The churn target is highly imbalanced. Accuracy alone is not sufficient; use precision, recall, F1, ROC-AUC, and PR-AUC together.")
        st.image(str(PROJECT_ROOT/"outputs"/"classification_confusion_matrix.png"), caption="Churn classification")
        st.image(str(PROJECT_ROOT/"outputs"/"regression_clv_actual_vs_predicted.png"), caption="CLV regression")
        st.image(str(PROJECT_ROOT/"outputs"/"regression_engagement_actual_vs_predicted.png"), caption="Engagement regression")


if __name__ == "__main__":
    main()
