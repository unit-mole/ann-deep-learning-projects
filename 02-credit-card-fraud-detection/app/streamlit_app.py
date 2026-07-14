"""Interactive Streamlit demo for ANN-based credit-card fraud detection."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from src.prediction_pipeline import FraudPredictionPipeline  # noqa: E402


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading ANN model and preprocessing artifacts...")
def load_pipeline() -> FraudPredictionPipeline:
    return FraudPredictionPipeline()


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    sample_path = PROJECT_ROOT / "data" / "sample_input.csv"
    return pd.read_csv(sample_path)


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def render_header() -> None:
    st.title("💳 Credit Card Fraud Detection")
    st.caption(
        "Artificial Neural Network • Batch risk scoring • Imbalanced classification"
    )
    st.info(
        "Portfolio demonstration only. This model is not a production banking "
        "control and should not be used for real financial decisions."
    )


def render_sidebar(pipeline: FraudPredictionPipeline) -> tuple[str, float]:
    st.sidebar.header("Prediction controls")
    mode = st.sidebar.radio(
        "Choose input method",
        ["Use demo sample", "Upload CSV", "Manual single transaction"],
    )

    default_threshold = min(
        max(float(pipeline.default_threshold), 0.01),
        1.00,
    )
    threshold = st.sidebar.slider(
        "Fraud decision threshold",
        min_value=0.01,
        max_value=1.00,
        value=float(round(default_threshold, 2)),
        step=0.01,
        help=(
            "Transactions with probability greater than or equal to this value "
            "are labeled FRAUD / RISK. Lower values usually increase recall but "
            "also increase false alerts."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Expected schema")
    st.sidebar.write("30 numeric features:")
    st.sidebar.code("Time, V1–V28, Amount")
    st.sidebar.caption(
        "An optional Class column may be included for demonstration evaluation."
    )
    return mode, threshold


def manual_input_form() -> pd.DataFrame | None:
    st.subheader("Manual transaction entry")
    st.warning(
        "V1–V28 are anonymized PCA features, so manual values are mainly useful "
        "for technical testing rather than intuitive business input."
    )

    defaults = {column: 0.0 for column in FEATURE_COLUMNS}
    defaults["Time"] = 94813.86
    defaults["Amount"] = 88.35
    editable = pd.DataFrame([defaults], columns=FEATURE_COLUMNS)

    edited = st.data_editor(
        editable,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
    )
    if st.button("Score manual transaction", type="primary"):
        return edited
    return None


def get_input_data(mode: str) -> pd.DataFrame | None:
    if mode == "Use demo sample":
        st.subheader("Curated demonstration sample")
        st.caption(
            "This file intentionally includes more fraud rows than the source "
            "dataset so both output classes are visible. It is not prevalence-representative."
        )
        return load_sample_data()

    if mode == "Upload CSV":
        st.subheader("Upload transaction data")
        uploaded_file = st.file_uploader(
            "Choose a CSV containing Time, V1–V28, and Amount",
            type=["csv"],
        )
        if uploaded_file is None:
            st.write("Upload a CSV to begin scoring.")
            return None
        try:
            return pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            return None

    return manual_input_form()


def render_summary(predictions: pd.DataFrame, threshold: float) -> None:
    total = len(predictions)
    predicted_fraud = int((predictions["prediction"] == 1).sum())
    predicted_safe = total - predicted_fraud
    fraud_rate = predicted_fraud / total if total else 0.0
    mean_probability = float(predictions["fraud_probability"].mean())

    metric_columns = st.columns(4)
    metric_columns[0].metric("Transactions scored", f"{total:,}")
    metric_columns[1].metric("SAFE", f"{predicted_safe:,}")
    metric_columns[2].metric("FRAUD / RISK", f"{predicted_fraud:,}")
    metric_columns[3].metric(
        "Predicted fraud rate",
        f"{fraud_rate:.2%}",
        help=f"Mean fraud probability: {mean_probability:.4%}",
    )
    st.caption(f"Decision threshold used: {threshold:.2f}")


def render_ground_truth_metrics(predictions: pd.DataFrame) -> None:
    if TARGET_COLUMN not in predictions.columns:
        return

    actual = pd.to_numeric(predictions[TARGET_COLUMN], errors="coerce")
    valid = actual.isin([0, 1])
    if not valid.all():
        st.warning(
            "The optional Class column contains values other than 0/1, so "
            "demonstration metrics were skipped."
        )
        return

    y_true = actual.astype(int)
    y_pred = predictions["prediction"].astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    with st.expander("Evaluation on uploaded labels", expanded=True):
        st.caption(
            "These metrics are available only because the current input includes "
            "the optional Class column."
        )
        columns = st.columns(4)
        columns[0].metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
        columns[1].metric(
            "Precision",
            f"{precision_score(y_true, y_pred, zero_division=0):.2%}",
        )
        columns[2].metric(
            "Recall",
            f"{recall_score(y_true, y_pred, zero_division=0):.2%}",
        )
        columns[3].metric(
            "F1-score",
            f"{f1_score(y_true, y_pred, zero_division=0):.2%}",
        )

        matrix_frame = pd.DataFrame(
            matrix,
            index=["Actual Safe", "Actual Fraud"],
            columns=["Predicted Safe", "Predicted Fraud"],
        )
        st.dataframe(matrix_frame, width="stretch")


def render_results(predictions: pd.DataFrame, threshold: float) -> None:
    render_summary(predictions, threshold)

    chart_data = (
        predictions["prediction_label"]
        .value_counts()
        .rename_axis("Prediction")
        .reset_index(name="Transactions")
    )
    st.subheader("Prediction distribution")
    st.bar_chart(chart_data, x="Prediction", y="Transactions")

    high_risk = predictions.sort_values(
        "fraud_probability", ascending=False
    ).copy()
    display_columns = [
        column
        for column in [
            "fraud_probability_pct",
            "prediction_label",
            "risk_band",
            "Amount",
            "Time",
            TARGET_COLUMN,
        ]
        if column in high_risk.columns
    ]

    st.subheader("Highest-risk transactions")
    st.dataframe(
        high_risk[display_columns].head(25),
        width="stretch",
        hide_index=True,
        column_config={
            "fraud_probability_pct": st.column_config.ProgressColumn(
                "Fraud probability",
                format="%.2f%%",
                min_value=0.0,
                max_value=100.0,
            )
        },
    )

    render_ground_truth_metrics(predictions)

    with st.expander("View complete scored dataset"):
        st.dataframe(predictions, width="stretch", hide_index=True)

    st.download_button(
        "Download predictions as CSV",
        data=dataframe_to_csv_bytes(predictions),
        file_name="fraud_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:
    render_header()

    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error(
            "The model artifacts could not be loaded. Confirm that the files in "
            f"models/ are present and dependencies are installed. Details: {exc}"
        )
        st.stop()

    mode, threshold = render_sidebar(pipeline)
    input_data = get_input_data(mode)

    if input_data is None:
        return

    if input_data.empty:
        st.warning("The selected CSV does not contain any transaction rows.")
        return

    st.subheader("Input preview")
    st.dataframe(input_data.head(20), width="stretch", hide_index=True)
    st.caption(f"Rows loaded: {len(input_data):,}")

    should_score = mode != "Manual single transaction" or input_data is not None
    if should_score:
        try:
            with st.spinner("Scoring transactions..."):
                predictions = pipeline.predict(
                    input_data,
                    threshold=threshold,
                )
        except (ValueError, TypeError, FileNotFoundError) as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Prediction failed unexpectedly: {exc}")
            st.stop()

        render_results(predictions, threshold)


if __name__ == "__main__":
    main()
