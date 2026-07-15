"""Interactive Streamlit demo for the Diabetes Risk Screening ANN."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, MODEL_DIR, OUTPUT_DIR  # noqa: E402
from src.prediction_pipeline import DiabetesPredictionPipeline  # noqa: E402

st.set_page_config(
    page_title="Diabetes Risk Screening ANN",
    page_icon="🩺",
    layout="wide",
)

DISCLAIMER = """
**Healthcare disclaimer:** This project is for educational and portfolio demonstration purposes only.  
It is **not a medical diagnostic tool**, and its predictions must not be used as medical advice.  
Consult a qualified healthcare professional for medical decisions.
"""


@st.cache_resource
def load_pipeline() -> DiabetesPredictionPipeline:
    return DiabetesPredictionPipeline(MODEL_DIR)


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "sample_input.csv")


@st.cache_data
def load_model_card() -> tuple[dict, pd.DataFrame]:
    metrics = json.loads((OUTPUT_DIR / "model_metrics.json").read_text())
    importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv")
    return metrics, importance


def display_score(result: pd.Series) -> None:
    probability = float(result["DiabetesRiskProbability"])
    category = str(result["RiskCategory"])
    st.subheader("Screening result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Diabetes risk probability", f"{probability:.1%}")
    col2.metric("Risk category", category)
    col3.metric("Model screening flag", result["ScreeningFlag"])
    st.progress(probability)
    st.info(str(result["Interpretation"]))
    st.caption(
        "Low, Medium, and High are transparent communication bands (<0.30, "
        "0.30–0.59, and ≥0.60). They are not clinical categories."
    )


st.title("Diabetes Risk Screening with an Artificial Neural Network")
st.write(
    "Estimate a model-based diabetes risk probability from eight health indicators, "
    "review risk bands, or score a CSV batch."
)
st.warning(DISCLAIMER)

try:
    pipeline = load_pipeline()
except Exception as error:
    st.error(f"The trained artifacts could not be loaded: {error}")
    st.stop()

manual_tab, batch_tab, model_tab = st.tabs(
    ["Individual screening", "Batch scoring", "Model card"]
)

with manual_tab:
    st.markdown("### Enter patient indicators")
    st.caption(
        "A value of 0 for glucose, blood pressure, skin thickness, insulin, or BMI "
        "is treated as missing and median-imputed by the saved training pipeline."
    )
    with st.form("manual_prediction"):
        left, right = st.columns(2)
        with left:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
            blood_pressure = st.number_input("Blood pressure (mm Hg)", min_value=0.0, max_value=160.0, value=72.0, step=1.0)
            skin_thickness = st.number_input("Skin thickness (mm)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
        with right:
            insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0, step=0.1)
            pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        submitted = st.form_submit_button("Estimate risk", type="primary")

    if submitted:
        patient = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
        }])
        display_score(pipeline.score(patient).iloc[0])
        with st.expander("Input used for scoring"):
            st.dataframe(patient, width="stretch", hide_index=True)

with batch_tab:
    st.markdown("### Score multiple records")
    source = st.radio(
        "Data source",
        ["Use included sample", "Upload CSV"],
        horizontal=True,
    )
    batch_data = None
    if source == "Use included sample":
        batch_data = load_sample_data()
    else:
        uploaded = st.file_uploader("Upload a CSV with the required eight columns", type=["csv"])
        if uploaded is not None:
            batch_data = pd.read_csv(uploaded)

    if batch_data is not None:
        st.write("Input preview")
        st.dataframe(batch_data.head(20), width="stretch", hide_index=True)
        if st.button("Run batch scoring", type="primary"):
            try:
                scored = pipeline.score(batch_data)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success(f"Scored {len(scored):,} records.")
                summary = (
                    scored["RiskCategory"]
                    .value_counts()
                    .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
                )
                st.bar_chart(summary)
                st.dataframe(scored, width="stretch", hide_index=True)
                st.download_button(
                    "Download scored CSV",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="diabetes_risk_predictions.csv",
                    mime="text/csv",
                )

with model_tab:
    metrics, importance = load_model_card()
    st.markdown("### Held-out test performance")
    metric_columns = st.columns(6)
    values = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1", metrics["f1_score"]),
        ("ROC-AUC", metrics["roc_auc"]),
        ("PR-AUC", metrics["pr_auc"]),
    ]
    for column, (label, value) in zip(metric_columns, values):
        column.metric(label, f"{value:.3f}")
    st.caption(
        f"Classification threshold: {metrics['classification_threshold']:.2f}. "
        "It was selected on the validation set by maximizing F2, which gives recall "
        "more weight than precision for this screening demonstration."
    )
    chart_data = importance.set_index("feature")["importance_mean_auc_decrease"]
    st.markdown("### Permutation importance")
    st.bar_chart(chart_data)
    st.caption(
        "Importance is measured as the mean decrease in test ROC-AUC after shuffling "
        "one input. It describes this model and dataset, not medical causality."
    )
    st.markdown("### Limitations")
    st.markdown(
        """- Small public benchmark dataset with limited demographic scope.
- No prospective or external clinical validation.
- The output is a model probability for portfolio demonstration, not a diagnosis."""
    )

st.divider()
st.warning(DISCLAIMER)
