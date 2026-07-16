from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, MODEL_DIR, OUTPUT_DIR
from src.prediction_pipeline import PredictionPipeline

st.set_page_config(
    page_title="Tabular Embedding ANN",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_pipeline() -> PredictionPipeline:
    return PredictionPipeline(MODEL_DIR)


@st.cache_data
def load_sample_input() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_input.csv")


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / "final_metrics_summary.csv")


pipeline = load_pipeline()
metadata = pipeline.metadata
encoders = pipeline.category_encoders

st.title("Tabular Deep Learning with Embeddings")
st.caption(
    "Interactive binary classification using learned categorical embeddings and scaled numerical features."
)

with st.sidebar:
    st.header("Decision settings")
    threshold = st.slider(
        "Classification threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(metadata.get("default_threshold", 0.50)),
        step=0.01,
        help="Move the threshold to explore the precision–recall trade-off.",
    )
    st.markdown("**Task:** Binary classification")
    st.markdown("**Model:** Multi-input embedding ANN")
    st.markdown("**Dataset:** 25,000 synthetic records")

single_tab, batch_tab, insight_tab = st.tabs(
    ["Single prediction", "Batch prediction", "Model insights"]
)

with single_tab:
    st.subheader("Enter one business profile")
    columns = st.columns(2)
    record: dict[str, object] = {}

    for index, feature in enumerate(metadata["categorical_features"]):
        encoder = encoders[feature]
        if hasattr(encoder, "categories_"):
            options = list(encoder.categories_)
        else:
            options = [str(value) for value in encoder.classes_]
        with columns[index % 2]:
            record[feature] = st.selectbox(
                feature.replace("_", " ").title(), options=options, key=f"single_{feature}"
            )

    ranges = metadata.get("numeric_ranges", {})
    defaults = metadata.get("numeric_defaults", {})
    for index, feature in enumerate(metadata["numerical_features"]):
        lower, upper = ranges.get(feature, [0.0, 1_000_000.0])
        default = defaults.get(feature, (lower + upper) / 2)
        with columns[index % 2]:
            if feature == "age":
                record[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=int(lower), max_value=int(upper), value=int(default), step=1,
                    key=f"single_{feature}",
                )
            else:
                step = 0.01 if feature == "credit_utilization" else 1.0
                record[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=float(lower), max_value=float(upper), value=float(default), step=step,
                    key=f"single_{feature}",
                )

    if st.button("Generate prediction", type="primary", width="stretch"):
        result = pipeline.predict(pd.DataFrame([record]), threshold=threshold).iloc[0]
        probability = float(result["prediction_probability"])
        metric_columns = st.columns(3)
        metric_columns[0].metric("Predicted outcome", result["prediction_label"])
        metric_columns[1].metric("Positive probability", f"{probability:.1%}")
        metric_columns[2].metric("Business bucket", result["business_bucket"])
        st.progress(probability)
        st.info(result["interpretation"])
        with st.expander("Input summary"):
            st.dataframe(pd.DataFrame([record]), width="stretch")

with batch_tab:
    st.subheader("Score a CSV file")
    source = st.radio(
        "Data source", ["Use included sample", "Upload CSV"], horizontal=True
    )
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV with the required feature columns", type=["csv"])
        batch_frame = pd.read_csv(uploaded) if uploaded is not None else None
    else:
        batch_frame = load_sample_input()

    if batch_frame is not None:
        st.write("Input preview")
        st.dataframe(batch_frame.head(20), width="stretch")
        st.caption(
            "Categorical features: " + ", ".join(metadata["categorical_features"])
            + "  |  Numerical features: " + ", ".join(metadata["numerical_features"])
        )

        if st.button("Score batch", type="primary", width="stretch"):
            try:
                scored = pipeline.predict(batch_frame, threshold=threshold)
            except ValueError as error:
                st.error(str(error))
            else:
                unknown_counts = scored.attrs.get("unknown_category_counts", {})
                total_unknowns = sum(unknown_counts.values())
                if total_unknowns:
                    details = ", ".join(
                        f"{feature}: {count}" for feature, count in unknown_counts.items() if count
                    )
                    st.warning(
                        "Unseen categorical values were mapped to documented fallback categories: "
                        + details
                    )

                summary_columns = st.columns(3)
                summary_columns[0].metric("Rows scored", f"{len(scored):,}")
                summary_columns[1].metric(
                    "Higher propensity",
                    f"{int((scored['predicted_class'] == 1).sum()):,}",
                )
                summary_columns[2].metric(
                    "Average probability", f"{scored['prediction_probability'].mean():.1%}"
                )

                chart_columns = st.columns(2)
                class_counts = scored["prediction_label"].value_counts().rename_axis("label").reset_index(name="count")
                with chart_columns[0]:
                    st.plotly_chart(
                        px.bar(class_counts, x="label", y="count", title="Prediction counts"),
                        width="stretch",
                    )
                with chart_columns[1]:
                    st.plotly_chart(
                        px.histogram(
                            scored,
                            x="prediction_probability",
                            nbins=20,
                            title="Prediction probability distribution",
                        ),
                        width="stretch",
                    )

                st.dataframe(scored, width="stretch")
                st.download_button(
                    "Download scored CSV",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="tabular_embedding_scored_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                )

with insight_tab:
    st.subheader("Why embeddings?")
    st.markdown(
        "Each categorical feature enters through its own input layer. The model converts category IDs "
        "into compact trainable vectors, flattens those vectors, concatenates them with scaled numerical "
        "features, and learns nonlinear interactions through dense layers."
    )

    metrics = load_metrics()
    st.write("Model comparison at threshold 0.50")
    display_metrics = metrics.copy()
    numeric_columns = [column for column in display_metrics.columns if column != "model"]
    display_metrics[numeric_columns] = display_metrics[numeric_columns].round(4)
    st.dataframe(display_metrics, hide_index=True, width="stretch")

    embedding_rows = []
    for feature in metadata["categorical_features"]:
        layer = pipeline.model.get_layer(f"{feature}_embedding")
        vocab_size, embedding_dimension = layer.get_weights()[0].shape
        embedding_rows.append(
            {
                "feature": feature,
                "vocabulary_size": vocab_size,
                "embedding_dimension": embedding_dimension,
            }
        )
    st.write("Learned embedding shapes")
    st.dataframe(pd.DataFrame(embedding_rows), hide_index=True, width="stretch")

    image_names = [
        "model_architecture.png",
        "training_curves.png",
        "confusion_matrix.png",
        "model_comparison.png",
        "embedding_visualization.png",
        "permutation_importance.png",
    ]
    for image_name in image_names:
        image_path = OUTPUT_DIR / image_name
        if image_path.exists():
            st.image(str(image_path), width="stretch")

st.divider()
st.caption(
    "Portfolio demo only. The dataset is synthetic and the target is a demonstration label, not a real-world risk decision."
)
