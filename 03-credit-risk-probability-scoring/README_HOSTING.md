# Hosting Guide

## Recommended option: Streamlit Community Cloud

This project is already a Streamlit application, the trained model is small enough to keep in Git, and Community Cloud can deploy directly from a GitHub repository. Use Python 3.11 for the closest compatibility with the included dependency ranges.

### Required repository files

- `app/streamlit_app.py`
- `requirements.txt`
- `models/final_credit_risk_ann_model.keras`
- `models/preprocessing_schema.json`
- `src/`
- `data/sample_input.csv`

### Deployment steps

1. Push the `ANN` repository to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Select **Create app** and choose the repository and branch.
4. Set the main file path to:
   `Credit_Risk_Probability_Scoring/app/streamlit_app.py`
5. Open **Advanced settings**, select Python 3.11, and deploy.
6. Watch the build logs. A successful deployment produces a public URL ending in `streamlit.app`.
7. Test manual scoring, sample scoring, CSV upload, the chart, and CSV download.
8. Add the live URL to this project's README, your resume, LinkedIn Featured section, and portfolio.

### Common fixes

- **Module not found:** verify the root `requirements.txt` is committed and the entrypoint path is correct.
- **Model not found:** confirm the `.keras` file and preprocessing schema are tracked by Git.
- **Memory/build issue:** remove optional development dependencies and deploy only `requirements.txt`.
- **Python incompatibility:** delete and redeploy the app with Python 3.11 selected.

## Alternative: Hugging Face Spaces

Hugging Face no longer treats Streamlit as a default built-in Space SDK, so use the included Dockerfile. Create a new **Docker Space**, copy the project files to the Space repository, keep port 8501, and push. This option is useful when you want the project displayed in an ML-focused public profile, but Streamlit Community Cloud is simpler for this repository.

## Local Docker test

```bash
docker build -t credit-risk-ann .
docker run --rm -p 8501:8501 credit-risk-ann
```

Then open `http://localhost:8501`.
