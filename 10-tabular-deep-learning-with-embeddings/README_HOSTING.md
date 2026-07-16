# Hosting Guide — Streamlit Community Cloud

Streamlit Community Cloud is the recommended first deployment option because this
project already uses Streamlit, the saved model is small, and the app can be deployed
directly from the GitHub monorepo without a Dockerfile.

## Files required for deployment

- `app/streamlit_app.py`
- `app/requirements.txt`
- `models/tabular_embedding_ann.keras`
- `models/label_encoders.pkl`
- `models/numeric_imputer.pkl`
- `models/numeric_scaler.pkl`
- `models/feature_metadata.json`
- `data/sample_input.csv`
- all modules under `src/`
- the selected visuals under `outputs/` used by the insights tab

The root `requirements.txt` is provided for convenient local installation. The
second copy at `app/requirements.txt` is intentionally beside the Streamlit
entrypoint so the hosted monorepo deployment resolves dependencies reliably.

## Recommended deployment settings

- Repository: `unit-mole/ann-deep-learning-projects`
- Branch: `main`
- App file: `10-tabular-deep-learning-with-embeddings/app/streamlit_app.py`
- Python version: `3.12`

Python 3.12 matches the intended project environment and is supported by the pinned
TensorFlow release.

## Add the project to the GitHub monorepo

1. Keep `10-tabular-deep-learning-with-embeddings/` in the repository root.
2. Keep `tabular-embedding-ci.yml` in the repository-level workflow folder:

```text
ann-deep-learning-projects/
├── .github/
│   └── workflows/
│       └── tabular-embedding-ci.yml
└── 10-tabular-deep-learning-with-embeddings/
```

Do not keep a second workflow copy inside the individual project folder.

3. From Windows Command Prompt, run:

```bat
cd /d "C:\path\to\ann-deep-learning-projects"
git add "10-tabular-deep-learning-with-embeddings" ".github\workflows\tabular-embedding-ci.yml"
git commit -m "Add tabular deep learning with embeddings project"
git pull --rebase origin main
git push origin main
```

## Deploy the app

1. Sign in to Streamlit Community Cloud with GitHub.
2. Select **Create app** and choose the repository and `main` branch.
3. Enter the complete app path shown above.
4. Open **Advanced settings** and select Python 3.12.
5. Deploy and review the build logs.
6. Test manual prediction, included sample scoring, CSV upload, and CSV download.
7. Choose a memorable custom subdomain, such as `tabular-embedding-ann`, if available.

The final shareable address will use the `streamlit.app` domain. Add that link to the
project README, the main repository project table, LinkedIn, your resume, and your
portfolio website.

## Local deployment check

```bat
cd /d "C:\path\to\10-tabular-deep-learning-with-embeddings"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app\streamlit_app.py
```

The included `run_local.bat` performs these steps automatically on Windows.

## Pre-deployment checklist

- The app starts without an exception.
- The sample file produces predictions between 0 and 1.
- An unseen category produces a warning rather than a crash.
- Batch results can be downloaded as CSV.
- All model and preprocessing artifacts are committed.
- The README live-demo placeholder is replaced after deployment.
- Two or three screenshots are saved under `images/`.

## Troubleshooting

- **Model load error:** verify `keras==3.14.0` and `tensorflow==2.21.0` are installed.
- **Missing artifact:** confirm all files under `models/` were committed to GitHub.
- **Pickle version warning:** use the pinned `scikit-learn==1.8.0`.
- **Dependency file not detected:** confirm `app/requirements.txt` is present beside the app entrypoint.
- **Wrong paths:** deploy using the complete nested path from the monorepo root.
- **Large first build:** TensorFlow is the largest dependency, so the initial deployment can take longer than a lightweight Streamlit app.

## Alternative: Hugging Face Spaces

Hugging Face Spaces is a valid alternative when a Docker-based deployment or more
runtime control is required. For this portfolio project, Streamlit Community Cloud
is simpler because it uses the existing Streamlit entrypoint and GitHub repository
without an additional Docker configuration.
