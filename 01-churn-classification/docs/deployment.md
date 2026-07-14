# Deployment Guide

## Recommended Platform

Streamlit Community Cloud is the simplest hosting option for this project.

## Local Test

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Choose **Create app**.
4. Select the repository and branch.
5. Set the entrypoint to:

```text
01-churn-classification/app.py
```

6. Open Advanced settings and use Python 3.12.
7. Deploy.
8. Add the generated app link to the project README, GitHub profile, resume,
   and LinkedIn project section.

## Docker

Build:

```bash
docker build -t ann-churn-classification .
```

Run:

```bash
docker run --rm -p 8501:8501 ann-churn-classification
```

Open:

```text
http://localhost:8501
```

## Common Deployment Issues

### Model does not load

Confirm these files are committed:

```text
models/model.h5
models/label_encoder_gender.pkl
models/onehot_encoder_geo.pkl
models/scaler.pkl
models/metadata.json
```

### Pickle version warning

The supplied preprocessing artifacts were inspected with scikit-learn 1.8.0.
The deployment requirements pin that version for compatibility.

### Memory or startup delay

TensorFlow is a large dependency. Initial cloud startup may be slower than a
basic Streamlit application. A later optimization can convert the model to
TensorFlow Lite or ONNX for lighter inference.
