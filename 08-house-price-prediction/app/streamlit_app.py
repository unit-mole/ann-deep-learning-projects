"""Interactive Streamlit demo for California median house value prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, METADATA_PATH, SAMPLE_INPUT_PATH
from src.prediction_pipeline import PredictionPipeline


st.set_page_config(
    page_title="ANN House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .small-note {font-size: 0.88rem; opacity: 0.82;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline() -> PredictionPipeline:
    return PredictionPipeline()


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_INPUT_PATH)


pipeline = load_pipeline()
metadata = pipeline.metadata
stats = metadata["feature_statistics"]

st.title("🏠 House Price Prediction with an Artificial Neural Network")
st.write(
    "Estimate the **median house value of a California census block group** "
    "from income, housing, occupancy, and geographic features."
)
st.info(
    "Scope note: this portfolio model uses the California Housing dataset. "
    "It is a block-group value estimator, not an individual-property appraisal."
)

single_tab, batch_tab, model_tab = st.tabs(
    ["Single prediction", "Batch prediction", "Model & results"]
)

with single_tab:
    st.subheader("Enter block-group characteristics")
    left, middle, right = st.columns(3)

    with left:
        med_inc = st.number_input(
            "Median income (tens of $10,000)",
            min_value=float(stats["MedInc"]["p01"]),
            max_value=float(stats["MedInc"]["p99"]),
            value=float(stats["MedInc"]["median"]),
            step=0.1,
            help="Example: 4.0 represents approximately $40,000 median income.",
        )
        house_age = st.number_input(
            "Median house age (years)",
            min_value=float(stats["HouseAge"]["min"]),
            max_value=float(stats["HouseAge"]["max"]),
            value=float(stats["HouseAge"]["median"]),
            step=1.0,
        )
        avg_rooms = st.number_input(
            "Average rooms per household",
            min_value=float(stats["AveRooms"]["p01"]),
            max_value=float(stats["AveRooms"]["p99"]),
            value=float(stats["AveRooms"]["median"]),
            step=0.1,
        )

    with middle:
        avg_bedrooms = st.number_input(
            "Average bedrooms per household",
            min_value=float(stats["AveBedrms"]["p01"]),
            max_value=float(stats["AveBedrms"]["p99"]),
            value=float(stats["AveBedrms"]["median"]),
            step=0.05,
        )
        population = st.number_input(
            "Block-group population",
            min_value=float(stats["Population"]["p01"]),
            max_value=float(stats["Population"]["p99"]),
            value=float(stats["Population"]["median"]),
            step=50.0,
        )
        avg_occupancy = st.number_input(
            "Average occupants per household",
            min_value=float(stats["AveOccup"]["p01"]),
            max_value=float(stats["AveOccup"]["p99"]),
            value=float(stats["AveOccup"]["median"]),
            step=0.1,
        )

    with right:
        latitude = st.number_input(
            "Latitude",
            min_value=float(stats["Latitude"]["min"]),
            max_value=float(stats["Latitude"]["max"]),
            value=float(stats["Latitude"]["median"]),
            step=0.01,
            format="%.2f",
        )
        longitude = st.number_input(
            "Longitude",
            min_value=float(stats["Longitude"]["min"]),
            max_value=float(stats["Longitude"]["max"]),
            value=float(stats["Longitude"]["median"]),
            step=0.01,
            format="%.2f",
        )

    input_values = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": avg_rooms,
        "AveBedrms": avg_bedrooms,
        "Population": population,
        "AveOccup": avg_occupancy,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    st.caption("Input preview")
    st.dataframe(pd.DataFrame([input_values]), use_container_width=True)

    if st.button("Predict median house value", type="primary"):
        result = pipeline.predict_one(input_values)

        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Predicted value", f"${result['prediction_usd']:,.0f}")
        metric_2.metric(
            "Empirical 80% range",
            f"${result['estimated_low_usd']:,.0f} – ${result['estimated_high_usd']:,.0f}",
        )
        metric_3.metric("Relative price category", str(result["category"]))

        st.markdown(result["interpretation"])
        st.caption(
            "The range is based on the held-out test-set absolute-error distribution; "
            "it is not a formal appraisal interval."
        )

        driver_frame = pd.DataFrame(result["local_drivers"])
        driver_frame["Estimated impact"] = driver_frame["impact_usd"].map(
            lambda value: f"${value:+,.0f}"
        )
        st.subheader("Local sensitivity drivers")
        st.dataframe(
            driver_frame[["display_name", "direction", "Estimated impact"]],
            use_container_width=True,
            hide_index=True,
        )

with batch_tab:
    st.subheader("Upload a CSV or use the included sample")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    use_sample = st.checkbox("Use preloaded sample data")

    input_frame = None
    if uploaded is not None:
        try:
            input_frame = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f"Could not read the uploaded CSV: {error}")
    elif use_sample:
        input_frame = load_sample_data()

    st.caption("Required columns: " + ", ".join(FEATURE_COLUMNS))

    if input_frame is not None:
        st.write("Input preview")
        st.dataframe(input_frame.head(25), use_container_width=True)

        missing = [column for column in FEATURE_COLUMNS if column not in input_frame.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        elif st.button("Run batch prediction", type="primary"):
            try:
                output = pipeline.predict_batch(input_frame)
            except Exception as error:
                st.error(f"Prediction failed: {error}")
            else:
                st.success(f"Generated {len(output):,} predictions.")
                st.dataframe(output.head(100), use_container_width=True)

                chart_data = output[["PredictedPriceUSD"]].reset_index()
                chart_data.rename(columns={"index": "Property row"}, inplace=True)
                st.bar_chart(chart_data.set_index("Property row"))

                figure, axis = plt.subplots(figsize=(7, 4))
                axis.hist(output["PredictedPriceUSD"], bins=min(30, max(5, len(output))))
                axis.set_xlabel("Predicted median house value (USD)")
                axis.set_ylabel("Number of rows")
                axis.set_title("Predicted Price Distribution")
                st.pyplot(figure)
                plt.close(figure)

                st.download_button(
                    "Download prediction CSV",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="house_price_predictions.csv",
                    mime="text/csv",
                )

with model_tab:
    st.subheader("Model performance")
    metrics = metadata["metrics_usd"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"${metrics['mae']:,.0f}")
    col2.metric("RMSE", f"${metrics['rmse']:,.0f}")
    col3.metric("R²", f"{metadata['metrics_target_units']['r2']:.3f}")
    col4.metric("MAPE", f"{metrics['mape_percent']:.1f}%")

    st.markdown(
        """
        **Architecture:** 8 inputs → Dense(128, ReLU) → Dropout(0.20) →
        Dense(64, ReLU) → Dense(1, linear).

        **Outlier policy:** observations were retained and diagnosed rather than
        deleted blindly, because extreme values can represent genuine geographic
        housing-market segments.
        """
    )

    image_columns = st.columns(2)
    output_dir = PROJECT_ROOT / "outputs"
    images = [
        ("Actual vs predicted", output_dir / "actual_vs_predicted.png"),
        ("Residual plot", output_dir / "residual_plot.png"),
        ("Feature importance", output_dir / "feature_importance.png"),
        ("Target distribution", output_dir / "price_distribution.png"),
    ]
    for index, (caption, path) in enumerate(images):
        if path.exists():
            image_columns[index % 2].image(str(path), caption=caption, use_container_width=True)

    with st.expander("Dataset and modeling limitations"):
        st.write(
            "- The target is median block-group house value, not an individual sale price.\n"
            "- The source target is capped near $500,001, limiting high-end predictions.\n"
            "- Inputs are aggregate census features; bedrooms and rooms are averages.\n"
            "- The empirical price range is not a calibrated probabilistic interval.\n"
            "- Predictions should support analysis, not replace a licensed appraisal."
        )
