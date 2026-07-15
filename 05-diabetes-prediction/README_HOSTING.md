# Hosting guide — Streamlit Community Cloud

Streamlit Community Cloud is recommended for this project because the interface is already written in Streamlit, the repository is public, and the platform can deploy a main file located inside a subdirectory of a monorepo.

## Required tracked files

- `app/streamlit_app.py`
- `app/requirements.txt`
- `requirements.txt`
- `models/diabetes_ann.keras`
- `models/preprocessor.joblib`
- `models/model_metadata.json`
- `outputs/model_metrics.json`
- `outputs/feature_importance.csv`
- `data/sample_input.csv`
- the `src/` package

No secrets or external APIs are required.

## 1. Test locally

```bat
cd /d "C:\Users\atripathi\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\ann-deep-learning-projects\05-diabetes-prediction"
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Confirm all three app tabs load, score the included sample, and download the result.

## 2. Commit the project and workflow

Run these commands from the root of `ann-deep-learning-projects`:

```bat
git status
git add 05-diabetes-prediction .github/workflows/diabetes-ann-ci.yml
git commit -m "Add GitHub-ready diabetes prediction ANN project"
git pull --rebase origin main
git push origin main
```

## 3. Deploy in Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Select **Create app** / **Deploy an app**.
3. Choose repository `unit-mole/ann-deep-learning-projects`.
4. Choose branch `main`.
5. Set the main file path to:

```text
05-diabetes-prediction/app/streamlit_app.py
```

6. Open advanced settings and select Python `3.12`.
7. Choose an available subdomain, for example:

```text
diabetes-risk-ann
```

8. Deploy and review the build logs.

The final shareable address will follow this pattern:

```text
https://diabetes-risk-ann.streamlit.app/
```

After deployment, replace the placeholder link in `README.md`, the project entry in the monorepo README, your resume, LinkedIn, and portfolio website.

## 4. Monorepo dependency behavior

Community Cloud searches the entrypoint directory first and then the repository root. Because this app lives at `05-diabetes-prediction/app/streamlit_app.py`, an identical dependency file is included at `05-diabetes-prediction/app/requirements.txt` for cloud deployment. The project-root `requirements.txt` remains the convenient file for local development and CI. Keep these two files synchronized.

## 5. Common troubleshooting

### Model artifact not found

Confirm the three files under `models/` were committed and that GitHub displays them. The app uses project-root-relative paths, not local Windows paths.

### TensorFlow installation failure

Confirm Python `3.12` was selected at initial deployment and the pinned `tensorflow-cpu` version remains in `requirements.txt`. Changing Python after deployment generally requires deleting and redeploying the app.

### App starts but batch scoring fails

Use the exact schema in `data/sample_input.csv`. Column spelling and capitalization must match.

### Memory or cold-start concerns

The model is small. It is cached with `st.cache_resource`, so it loads once per app process rather than on every interaction.

### Repository is private

Community Cloud can connect to authorized private GitHub repositories, but a public portfolio repository is simpler for recruiters to review.

## Optional Hugging Face Spaces route

A Hugging Face Space can also host Streamlit-compatible apps, but Streamlit Community Cloud is the cleaner first choice for this Streamlit-native monorepo. Use a Space only when you want the project grouped with model cards or other Hugging Face assets.
