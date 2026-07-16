# Hosting Guide — Streamlit Community Cloud

## Recommended platform

Use **Streamlit Community Cloud** for this project because the application is
already written in Streamlit, the repository can be connected directly, and
the resulting public URL is easy to share on GitHub, LinkedIn, a résumé, or a
portfolio website.

## Files required for deployment

```text
08-house-price-prediction/
├── app/
│   ├── streamlit_app.py
│   └── requirements.txt
├── src/
├── models/
│   ├── house_price_ann.keras
│   ├── house_price_scaler.pkl
│   └── model_metadata.json
├── data/
│   └── sample_input.csv
├── outputs/
└── requirements.txt
```

The full training dataset is not required by the deployed app, but it is
included for reproducible training.

## GitHub preparation

From the root of the `ann-deep-learning-projects` repository:

```bat
git add "08-house-price-prediction" ".github/workflows/house-price-ann-ci.yml"
git commit -m "Add GitHub-ready ANN house price prediction project"
git pull --rebase origin main
git push origin main
```

## Deploy on Streamlit Community Cloud

1. Sign in using the GitHub account that can access the repository.
2. Create a new app and select `unit-mole/ann-deep-learning-projects`.
3. Select the `main` branch.
4. Set the app file path to:

```text
08-house-price-prediction/app/streamlit_app.py
```

5. Open **Advanced settings** and select Python **3.12**.
6. Deploy the app.
7. Review build logs if dependency installation fails.
8. Test manual prediction, sample batch prediction, CSV download, and model
   performance images.
9. Copy the public `streamlit.app` URL and replace the placeholder in `README.md`.

## Suggested public URL label

```text
Live Demo: ANN House Price Prediction
```

## Troubleshooting

### TensorFlow installation takes time or memory

Keep the `.keras` model small, avoid SHAP in the deployed environment, and do
not retrain inside Streamlit. The app only loads artifacts and performs
inference.

### Scikit-learn pickle warning

The saved scaler was created with scikit-learn 1.8.0, so the requirements file
pins that version.

### App cannot find artifacts

Do not use machine-specific absolute paths. The supplied application resolves
all files relative to the project folder.

### App works locally but not online

Confirm that the following files are committed:

```text
models/house_price_ann.keras
models/house_price_scaler.pkl
models/model_metadata.json
data/sample_input.csv
outputs/*.png
```

## Alternative: Hugging Face Spaces

Hugging Face no longer provides Streamlit as a built-in Spaces SDK. A Streamlit
app now requires the **Docker SDK** and the Streamlit Docker template. That adds
Docker configuration and maintenance, so Streamlit Community Cloud is the
cleaner primary option for this portfolio project.
