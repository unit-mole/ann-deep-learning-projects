"""Interactive Streamlit demo for the MNIST dense ANN."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_preprocessing import PreprocessedDigit, preprocess_digit_image
from src.prediction_pipeline import DigitPrediction, load_digit_model, predict_digit

MODEL_PATH = PROJECT_ROOT / 'models' / 'digit_recognition_model.keras'
METRICS_PATH = PROJECT_ROOT / 'outputs' / 'model_metrics.json'
SAMPLE_DIR = PROJECT_ROOT / 'data' / 'sample_digits'


@st.cache_resource(show_spinner='Loading the neural network...')
def get_model():
    return load_digit_model(MODEL_PATH)


@st.cache_data
def get_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding='utf-8'))


def probability_frame(prediction: DigitPrediction) -> pd.DataFrame:
    frame = pd.DataFrame({
        'Digit': list(range(10)),
        'Probability': prediction.probabilities,
    })
    frame['Confidence'] = frame['Probability'].map(lambda value: f'{value:.2%}')
    frame['Rank'] = frame['Probability'].rank(method='first', ascending=False).astype(int)
    return frame.sort_values('Probability', ascending=False).reset_index(drop=True)


def render_header() -> None:
    st.title('Handwritten Digit Recognition')
    st.caption('Dense ANN • MNIST • 10-class image classification')
    st.write(
        'Upload a clear image of one handwritten digit or choose a built-in MNIST sample. '
        'The app converts it to the same 28×28 white-on-black format used during training.'
    )


def select_input_image() -> Image.Image | None:
    source = st.radio('Choose an input source', ['Upload an image', 'Use a sample'], horizontal=True)
    if source == 'Upload an image':
        uploaded = st.file_uploader(
            'Upload PNG, JPG, JPEG, or BMP',
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help='Use one centered digit with strong foreground/background contrast.',
        )
        if uploaded is None:
            return None
        return Image.open(io.BytesIO(uploaded.getvalue()))

    samples = sorted(SAMPLE_DIR.glob('digit_*.png'))
    if not samples:
        st.error('No sample images were found.')
        return None
    selected = st.selectbox('Select a sample digit', samples, format_func=lambda path: path.stem.replace('_', ' ').title())
    return Image.open(selected)


def render_preprocessing(preprocessed: PreprocessedDigit) -> None:
    st.subheader('Preprocessing preview')
    first, second, third = st.columns(3)
    with first:
        st.image(preprocessed.original, caption='Original input', use_container_width=True)
    with second:
        st.image(preprocessed.grayscale, caption='Grayscale', use_container_width=True)
    with third:
        st.image(
            preprocessed.centered_28x28.resize((280, 280), Image.Resampling.NEAREST),
            caption='Centered 28×28 model input',
            use_container_width=True,
        )
    inversion = 'Yes — converted from black-on-white to MNIST format.' if preprocessed.was_inverted else 'No.'
    st.caption(f'Automatic inversion applied: {inversion}')


def render_prediction(prediction: DigitPrediction) -> None:
    st.subheader('Prediction')
    col1, col2, col3 = st.columns(3)
    col1.metric('Predicted digit', str(prediction.predicted_digit))
    col2.metric('Confidence', f'{prediction.confidence:.2%}')
    col3.metric('Runner-up', f'{prediction.second_digit} ({prediction.second_confidence:.2%})')

    if prediction.confidence < 0.60:
        st.warning('Low confidence: try a clearer, larger, and more centered digit.')
    elif prediction.confidence < 0.85:
        st.info('Moderate confidence: the image may differ from typical MNIST handwriting.')
    else:
        st.success('High-confidence prediction.')

    probabilities = probability_frame(prediction)
    chart_data = probabilities.sort_values('Digit')
    figure = px.bar(
        chart_data,
        x='Digit',
        y='Probability',
        text=chart_data['Probability'].map(lambda value: f'{value:.1%}'),
        title='Probability distribution across digits 0–9',
    )
    figure.update_layout(yaxis_tickformat='.0%', xaxis={'dtick': 1}, showlegend=False)
    figure.update_traces(textposition='outside', cliponaxis=False)
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        probabilities[['Rank', 'Digit', 'Confidence']],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        'Download probabilities as CSV',
        data=probabilities.to_csv(index=False).encode('utf-8'),
        file_name='digit_probabilities.csv',
        mime='text/csv',
    )


def render_model_performance() -> None:
    metrics = get_metrics()
    if not metrics:
        st.info('Run the evaluation pipeline to generate model metrics.')
        return
    first, second, third, fourth = st.columns(4)
    first.metric('Test accuracy', f"{metrics.get('test_accuracy', 0):.2%}")
    second.metric('Macro F1', f"{metrics.get('macro_f1', 0):.2%}")
    third.metric('Correct', f"{metrics.get('correct_predictions', 0):,}")
    fourth.metric('Errors', f"{metrics.get('misclassified_predictions', 0):,}")

    st.markdown('**Architecture:** 784 inputs → Dense 384 → Dropout → Dense 192 → Dropout → Softmax 10')
    st.caption(
        'Confidence is the model’s softmax probability, not a guarantee of correctness. '
        'Images that differ from MNIST can be confidently misclassified.'
    )


def main() -> None:
    st.set_page_config(
        page_title='Handwritten Digit Recognition ANN',
        page_icon='🔢',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    render_header()
    with st.sidebar:
        st.header('Model card')
        render_model_performance()
        st.divider()
        st.markdown(
            '**Best upload practices**\n\n'
            '- One digit only\n'
            '- Strong contrast\n'
            '- Minimal background clutter\n'
            '- Digit occupies most of the image\n'
            '- PNG or JPG format'
        )

    try:
        image = select_input_image()
        if image is None:
            st.info('Upload an image or choose a sample to begin.')
            return
        preprocessed = preprocess_digit_image(image)
        render_preprocessing(preprocessed)
        prediction = predict_digit(get_model(), preprocessed.model_input)
        render_prediction(prediction)
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
    except Exception as exc:  # Deployment-friendly user message; full details remain in logs.
        st.error('The prediction could not be completed. Check the app logs and dependency versions.')
        st.exception(exc)


if __name__ == '__main__':
    main()
