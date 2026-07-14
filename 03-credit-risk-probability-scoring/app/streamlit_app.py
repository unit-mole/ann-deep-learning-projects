from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSIFICATION_THRESHOLD, DATA_DIR
from src.prediction_pipeline import load_model, load_preprocessor, score_applicants

st.set_page_config(page_title="Credit Risk ANN Scoring", page_icon="💳", layout="wide")


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_model(), load_preprocessor()


@st.cache_data(show_spinner=False)
def get_sample_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_input.csv")


def manual_input() -> pd.DataFrame:
    st.subheader("Applicant details")
    left, middle, right = st.columns(3)
    with left:
        age = st.number_input("Age", 18, 85, 34)
        annual_income = st.number_input("Annual income ($)", 1_000.0, 1_000_000.0, 52_000.0, step=1_000.0)
        loan_amount = st.number_input("Loan amount ($)", 500.0, 500_000.0, 22_000.0, step=500.0)
        interest_rate = st.number_input("Interest rate (%)", 0.0, 60.0, 16.8, step=0.1)
        employment_length = st.number_input("Employment length (years)", 0.0, 50.0, 3.0, step=1.0)
        credit_history_years = st.number_input("Credit history (years)", 0.0, 60.0, 4.5, step=0.5)
    with middle:
        revolving_utilization = st.number_input("Revolving utilization (%)", 0.0, 100.0, 72.0, step=1.0)
        dti = st.number_input("Debt-to-income ratio (%)", 0.0, 100.0, 28.0, step=1.0)
        delinquency_count = st.number_input("Delinquencies", 0, 30, 1)
        inquiry_count = st.number_input("Recent credit inquiries", 0, 30, 3)
        open_accounts = st.number_input("Open accounts", 0, 60, 6)
        existing_loans = st.number_input("Existing loans", 0, 30, 3)
        collateral_value = st.number_input("Collateral value ($)", 0.0, 1_000_000.0, 2_500.0, step=500.0)
    with right:
        home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        purpose = st.selectbox("Loan purpose", ["debt_consolidation", "home_improvement", "education", "small_business", "medical", "auto", "vacation"], index=3)
        grade = st.selectbox("Credit grade", ["A", "B", "C", "D", "E", "F", "G"], index=4)
        region = st.selectbox("Region", ["North", "South", "East", "West"], index=3)
        verification_status = st.selectbox("Verification status", ["verified", "not_verified", "source_verified"], index=1)
    return pd.DataFrame([{
        "age": age, "annual_income": annual_income, "loan_amount": loan_amount,
        "interest_rate": interest_rate, "employment_length": employment_length,
        "credit_history_years": credit_history_years, "revolving_utilization": revolving_utilization,
        "dti": dti, "delinquency_count": delinquency_count, "inquiry_count": inquiry_count,
        "open_accounts": open_accounts, "existing_loans": existing_loans,
        "collateral_value": collateral_value, "home_ownership": home_ownership,
        "purpose": purpose, "grade": grade, "region": region,
        "verification_status": verification_status,
    }])


def render_results(scored: pd.DataFrame) -> None:
    st.subheader("Scoring result")
    first = scored.iloc[0]
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Default-risk probability", f"{first['risk_probability']:.1%}")
    metric2.metric("Risk category", first["risk_category"])
    metric3.metric("Predicted class", "Higher default risk" if int(first["predicted_default"]) else "Lower default risk")
    st.info(first["decision_recommendation"])

    counts = scored["risk_category"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0).reset_index()
    counts.columns = ["Risk Category", "Applicants"]
    chart = px.bar(counts, x="Risk Category", y="Applicants", text="Applicants", title="Risk category distribution")
    st.plotly_chart(chart, width="stretch")

    display_columns = [
        "risk_probability", "risk_category", "predicted_default",
        "decision_recommendation", "review_priority",
    ]
    remaining = [column for column in scored.columns if column not in display_columns]
    st.dataframe(scored[display_columns + remaining], width="stretch", hide_index=True)
    st.download_button(
        "Download scored CSV", data=scored.to_csv(index=False).encode("utf-8"),
        file_name="credit_risk_scored_output.csv", mime="text/csv",
    )


st.title("Credit Risk Probability Scoring with ANN")
st.caption("Portfolio demonstration using a synthetic banking dataset. Not for real lending decisions.")
st.write(
    "Estimate probability of default, assign an operational risk band, and produce a review recommendation. "
    f"The binary decision threshold is {CLASSIFICATION_THRESHOLD:.2f}; risk bands are separate business-policy layers."
)

mode = st.radio("Choose an input method", ["Manual entry", "Upload CSV", "Use sample data"], horizontal=True)
try:
    model, preprocessor = get_artifacts()
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()

input_data = None
if mode == "Manual entry":
    input_data = manual_input()
    run = st.button("Score applicant", type="primary")
elif mode == "Upload CSV":
    uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])
    run = uploaded is not None and st.button("Score uploaded file", type="primary")
    if uploaded is not None:
        input_data = pd.read_csv(uploaded)
        st.dataframe(input_data.head(25), width="stretch")
else:
    input_data = get_sample_data()
    st.dataframe(input_data, width="stretch")
    run = st.button("Score sample applicants", type="primary")

if run and input_data is not None:
    try:
        with st.spinner("Scoring applicants..."):
            scored_data = score_applicants(input_data, model=model, preprocessor=preprocessor)
        render_results(scored_data)
    except Exception as exc:
        st.exception(exc)

with st.expander("Input schema and interpretation"):
    st.markdown(
        "The CSV must include the raw applicant columns shown in `data/sample_input.csv`. "
        "`monthly_income`, `loan_to_income`, and `collateral_ratio` are derived automatically when absent. "
        "Unknown categorical levels are encoded as all-zero values for that category group."
    )
