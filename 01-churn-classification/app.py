from __future__ import annotations

from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import DATA_DIR
from src.inference import ChurnPredictor
from src.validation import InputValidationError

st.set_page_config(
    page_title="ANN Churn Classification",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .portfolio-note {
        padding: 0.9rem 1rem;
        border: 1px solid #d9e2f0;
        border-radius: 0.6rem;
        background: #f7f9fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> ChurnPredictor:
    return ChurnPredictor()


def probability_gauge(probability: float, threshold: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Predicted churn probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=20))
    return figure


def metric_value(metadata: dict, key: str) -> str:
    value = metadata.get("evaluation", {}).get(key)
    return "N/A" if value is None else f"{float(value):.3f}"


st.title("ANN Churn Classification")
st.caption(
    "Interactive portfolio demo for estimating the probability that a bank "
    "customer will leave."
)

try:
    predictor = load_predictor()
except Exception as error:
    st.error(
        "The model could not be loaded. Install the pinned dependencies and "
        "confirm that the files under `models/` are present."
    )
    st.exception(error)
    st.stop()

metadata = predictor.metadata
default_threshold = float(metadata.get("default_threshold", 0.5))

with st.sidebar:
    st.header("Decision Settings")
    threshold = st.slider(
        "Classification threshold",
        min_value=0.10,
        max_value=0.99,
        value=float(default_threshold),
        step=0.01,
        help=(
            "Lowering the threshold identifies more potential churners but "
            "may increase false alarms."
        ),
    )
    st.divider()
    st.subheader("Artifact mode")
    st.code(predictor.mode)
    st.markdown(
        """
        <div class="portfolio-note">
        This is an educational portfolio demonstration. Predictions should
        not be used as the sole basis for customer decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

single_tab, batch_tab, model_tab = st.tabs(
    ["Single Customer", "Batch Prediction", "Model Information"]
)

with single_tab:
    st.subheader("Customer Profile")

    with st.form("single_customer_form"):
        left, middle, right = st.columns(3)

        with left:
            credit_score = st.number_input(
                "Credit score", min_value=300, max_value=900, value=650
            )
            geography = st.selectbox(
                "Geography", ["France", "Germany", "Spain"]
            )
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input(
                "Age", min_value=18, max_value=100, value=40
            )

        with middle:
            tenure = st.number_input(
                "Tenure (years)", min_value=0, max_value=10, value=5
            )
            balance = st.number_input(
                "Account balance",
                min_value=0.0,
                max_value=300000.0,
                value=75000.0,
                step=1000.0,
            )
            number_of_products = st.number_input(
                "Number of products", min_value=1, max_value=4, value=2
            )

        with right:
            has_credit_card = st.selectbox(
                "Has credit card", ["Yes", "No"]
            )
            is_active_member = st.selectbox(
                "Active member", ["Yes", "No"]
            )
            estimated_salary = st.number_input(
                "Estimated salary",
                min_value=0.0,
                max_value=250000.0,
                value=100000.0,
                step=1000.0,
            )

        submitted = st.form_submit_button(
            "Estimate Churn Risk", type="primary", use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame(
            [
                {
                    "CreditScore": credit_score,
                    "Geography": geography,
                    "Gender": gender,
                    "Age": age,
                    "Tenure": tenure,
                    "Balance": balance,
                    "NumOfProducts": number_of_products,
                    "HasCrCard": 1 if has_credit_card == "Yes" else 0,
                    "IsActiveMember": 1 if is_active_member == "Yes" else 0,
                    "EstimatedSalary": estimated_salary,
                }
            ]
        )

        try:
            result = predictor.predict(customer, threshold=threshold).iloc[0]
        except InputValidationError as error:
            st.error(str(error))
        else:
            probability = float(result["ChurnProbability"])
            output_left, output_right = st.columns([1, 1.25])

            with output_left:
                st.plotly_chart(
                    probability_gauge(probability, threshold),
                    use_container_width=True,
                )

            with output_right:
                st.metric(
                    "Prediction",
                    result["PredictionLabel"],
                )
                st.metric("Risk band", result["RiskBand"])
                st.metric("Decision threshold", f"{threshold:.0%}")

                if int(result["PredictedChurn"]) == 1:
                    st.warning(
                        "The model flags this profile for proactive retention "
                        "review."
                    )
                else:
                    st.success(
                        "The model does not flag this profile at the selected "
                        "threshold."
                    )

                st.caption(
                    "A churn score is a prioritization signal—not a causal "
                    "explanation or guaranteed outcome."
                )

with batch_tab:
    st.subheader("Score Multiple Customers")
    sample_path = DATA_DIR / "sample" / "batch_prediction_sample.csv"
    sample_bytes = sample_path.read_bytes()

    st.download_button(
        "Download sample input CSV",
        data=sample_bytes,
        file_name="batch_prediction_sample.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV using the same columns as the sample file",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            batch_frame = pd.read_csv(uploaded_file)
            batch_result = predictor.predict(
                batch_frame,
                threshold=threshold,
            )
        except (InputValidationError, ValueError) as error:
            st.error(str(error))
        except Exception as error:
            st.error("The uploaded file could not be processed.")
            st.exception(error)
        else:
            predicted_count = int(batch_result["PredictedChurn"].sum())
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Rows scored", f"{len(batch_result):,}")
            col_b.metric("Flagged for churn", f"{predicted_count:,}")
            col_c.metric(
                "Flagged rate",
                f"{predicted_count / len(batch_result):.1%}",
            )

            st.dataframe(
                batch_result.sort_values(
                    "ChurnProbability", ascending=False
                ),
                use_container_width=True,
            )

            csv_buffer = StringIO()
            batch_result.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download prediction results",
                data=csv_buffer.getvalue(),
                file_name="churn_predictions.csv",
                mime="text/csv",
                type="primary",
            )

with model_tab:
    st.subheader("Current Model Snapshot")

    metric_columns = st.columns(6)
    metric_columns[0].metric(
        "Accuracy", metric_value(metadata, "accuracy")
    )
    metric_columns[1].metric(
        "Churn precision", metric_value(metadata, "precision_churn")
    )
    metric_columns[2].metric(
        "Churn recall", metric_value(metadata, "recall_churn")
    )
    metric_columns[3].metric(
        "Churn F1", metric_value(metadata, "f1_churn")
    )
    metric_columns[4].metric(
        "ROC-AUC", metric_value(metadata, "roc_auc")
    )
    metric_columns[5].metric(
        "PR-AUC", metric_value(metadata, "pr_auc")
    )

    st.markdown(
        """
        ### ANN Architecture

        ```text
        12 processed inputs
                ↓
        Dense layer: 64 ReLU units
                ↓
        Dense layer: 16 ReLU units
                ↓
        Sigmoid churn probability
        ```
        """
    )

    if predictor.mode == "legacy_notebook_artifacts":
        st.warning(
            "These metrics come from the original notebook artifacts. The "
            "test set was also used as validation data during final early "
            "stopping, so this should be treated as a legacy benchmark. Run "
            "`python -m src.train` for the cleaner stratified benchmark."
        )

    st.markdown(
        """
        ### Important Interpretation

        - The dataset is imbalanced: approximately 20% of records are churn cases.
        - Accuracy alone can hide missed churners.
        - Churn recall measures how many actual churners are identified.
        - The threshold should reflect the cost of missed churn versus unnecessary outreach.
        - Probability calibration, fairness, and data drift should be checked before operational use.
        """
    )
