# Hosting Guide — Streamlit Community Cloud

Streamlit Community Cloud is the recommended option for this project because the app is already written in Streamlit, the model artifacts are small enough to remain in GitHub, and deployment can be connected directly to the repository.

## Required repository files

```text
06-dynamic-pricing-optimization/
├── app/streamlit_app.py
├── models/*.keras
├── models/numeric_scaler.joblib
├── models/model_metadata.json
├── data/sample_input.csv
├── src/
├── requirements.txt
└── .python-version
```

No secrets are required for this demo.

## 1. Test locally

From the project folder:

### Windows Command Prompt

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### macOS/Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open the local URL shown by Streamlit and verify both the single-product and batch tabs.

## 2. Push the project to GitHub

From the monorepo root:

```bash
git add "06-dynamic-pricing-optimization" ".github/workflows/dynamic-pricing-ci.yml"
git commit -m "Add ANN dynamic pricing optimization project"
git pull --rebase origin main
git push origin main
```

Confirm that the `.keras` models and `numeric_scaler.joblib` appear in GitHub. The transient training payloads are intentionally ignored.

## 3. Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Select **Create app**.
3. Choose the GitHub repository and the `main` branch.
4. Set the app entrypoint to:

```text
06-dynamic-pricing-optimization/app/streamlit_app.py
```

5. In advanced settings, select Python 3.12 if the interface offers a Python version selector.
6. Deploy the app and inspect the build logs for dependency or path errors.
7. Test a manual recommendation, the preloaded sample batch, and CSV download.

The shareable result will use a Streamlit-hosted URL ending in `.streamlit.app`. Add that URL to the project README, the main repository README, LinkedIn, and your resume.

## 4. Update the README badge and demo link

Replace the placeholder in `README.md`:

```text
https://YOUR-APP-NAME.streamlit.app
```

Commit and push the README update.

## Troubleshooting

- **Module not found:** verify `requirements.txt` is at the project root and the app entrypoint is correct.
- **Model not found:** confirm the two `.keras` files, scaler, and metadata file were committed.
- **Memory issue:** keep batch runs below 500 rows in the public demo.
- **Slow first run:** model loading is cached; subsequent interactions should be faster.
- **Backend error:** do not remove `torch` or `keras` from `requirements.txt`; the app explicitly selects the Keras PyTorch backend before importing Keras.

## Alternative: Hugging Face Spaces

Hugging Face Spaces can also host Streamlit-style apps, but it generally requires additional Space configuration. For this repository, Streamlit Community Cloud is the simpler and more direct deployment route.
