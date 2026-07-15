# Hosting Guide — Customer Lifetime Value Forecasting

## Recommended Platform

Use **Streamlit Community Cloud** for this project.

Why it is the best fit:

- The application is already written in Streamlit.
- It connects directly to a GitHub repository and branch.
- Commits automatically trigger app updates.
- No Dockerfile or server configuration is required.
- The final public URL is easy to share on GitHub, LinkedIn, a resume, or a portfolio.

Hugging Face Spaces remains possible, but its built-in Streamlit SDK was deprecated in April 2025; a new Streamlit Space now requires a Docker template. That adds unnecessary complexity for this portfolio app.

## Files Required for Deployment

Keep these files committed:

```text
04-customer-lifetime-value-forecasting/
├── app/
│   ├── streamlit_app.py
│   └── requirements.txt
├── data/sample_input.csv
├── models/
│   ├── clv_ann_model.keras
│   ├── label_encoders.pkl
│   ├── numeric_scaler.pkl
│   └── model_metadata.json
├── outputs/model_metrics.json
├── outputs/*.png
└── src/*.py
```

The trained model is approximately 1.1 MB, so normal Git is sufficient; Git LFS is not required.

## Step 1 — Test Locally

From the project folder:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Verify all four tabs:

1. Overview
2. Single Customer
3. Batch Scoring
4. Model Performance

Also test the included `data/sample_input.csv` and download the scored result.

## Step 2 — Place the Project in the Monorepo

Use this exact folder location:

```text
ann-deep-learning-projects/
└── 04-customer-lifetime-value-forecasting/
```

Move the supplied CI file to the monorepo workflow directory:

```text
From: 04-customer-lifetime-value-forecasting/ci/clv-ann-ci.yml
To:   .github/workflows/clv-ann-ci.yml
```

## Step 3 — Commit and Push

Run from the root of `ann-deep-learning-projects`:

```bat
git status
git add 04-customer-lifetime-value-forecasting .github/workflows/clv-ann-ci.yml
git commit -m "Add customer lifetime value forecasting ANN project"
git pull --rebase origin main
git push origin main
```

## Step 4 — Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Select **Create app**.
3. Choose **Yup, I have an app**.
4. Set the repository to `unit-mole/ann-deep-learning-projects`.
5. Set the branch to `main`.
6. Set the entrypoint file to:

```text
04-customer-lifetime-value-forecasting/app/streamlit_app.py
```

7. Open **Advanced settings** and select **Python 3.12**.
8. No secrets are required.
9. Choose a memorable subdomain, such as `clv-forecasting-ann`, if available.
10. Deploy the app and review the build logs.

The dependency file is placed next to the entrypoint at `app/requirements.txt`, which is one of the locations Streamlit Community Cloud searches.

## Step 5 — Share the Final Link

The final address will use this pattern:

```text
https://<your-subdomain>.streamlit.app
```

Add it to:

- The `Live demo` line in `README.md`
- Your GitHub repository About section
- Your resume project entry
- LinkedIn Featured or Projects
- Your personal portfolio website

After deployment, replace the README placeholder with the real URL and add the official Streamlit badge:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-subdomain>.streamlit.app)
```

## Build Troubleshooting

### TensorFlow or Keras model-load error

Confirm Streamlit Cloud is using Python 3.12 and that both files are committed:

```text
models/clv_ann_model.keras
models/model_metadata.json
```

The saved artifact reports Keras 3.14.0, which is pinned in `requirements.txt`.

### `ModuleNotFoundError: src`

Use the provided `app/streamlit_app.py` unchanged. It adds the project root to `sys.path` before importing project modules.

### File not found

Do not use Windows backslashes inside application paths. The code uses `pathlib`, which is portable across Windows and Linux.

### Slow first launch

TensorFlow creates a heavier cold start than a simple analytics app. Subsequent reruns should use Streamlit's cached model resource.

### Dependency changes not reflected

Commit and push the updated `requirements.txt`. Streamlit Community Cloud detects dependency changes and rebuilds the environment.

## Alternative — Hugging Face Spaces

Only choose Hugging Face Spaces when you specifically want the demo on your Hugging Face profile. Because the built-in Streamlit SDK has been deprecated, create a **Docker Space**, install the same dependencies, expose port 8501, and start:

```text
streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

For this project, Streamlit Community Cloud remains simpler and more recruiter-friendly.
