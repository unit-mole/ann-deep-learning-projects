# Hosting guide — Streamlit Community Cloud

Streamlit Community Cloud is the selected hosting platform for this project because the interface is Streamlit-native, the repository is public, and the application entrypoint is located inside a subdirectory of the ANN monorepo.

## Live deployment

The application has been deployed successfully and is publicly accessible without requiring a Streamlit sign-in.

**Live application:**  
[https://ann-deep-learning-projects-bczyq9q5aa8eqbvqskqyar.streamlit.app/](https://ann-deep-learning-projects-bczyq9q5aa8eqbvqskqyar.streamlit.app/)

Deployment configuration:

- **Repository:** `unit-mole/ann-deep-learning-projects`
- **Branch:** `main`
- **Main file path:** `05-diabetes-prediction/app/streamlit_app.py`
- **Python version:** `3.12`
- **Hosting platform:** Streamlit Community Cloud
- **Visibility:** Public
- **Deployment status:** Active

## Required tracked files

The deployed app depends on the following repository files:

- `app/streamlit_app.py`
- `app/requirements.txt`
- `requirements.txt`
- `models/diabetes_ann.keras`
- `models/preprocessor.joblib`
- `models/model_metadata.json`
- `outputs/model_metrics.json`
- `outputs/feature_importance.csv`
- `data/sample_input.csv`
- the complete `src/` package

No secrets, API keys, databases, or external services are required.

## 1. Test locally

```bat
cd /d "C:\Users\atripathi\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\ann-deep-learning-projects\05-diabetes-prediction"
py -3.12 -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m streamlit run app\streamlit_app.py
```

Before publishing changes, confirm that:

- all automated tests pass;
- the Individual Screening tab produces a result;
- the included batch sample is scored successfully;
- CSV output can be downloaded;
- the Model Card tab loads its metrics and feature-importance chart.

## 2. Commit and push updates

Run these commands from the root of `ann-deep-learning-projects`:

```bat
git status
git add "05-diabetes-prediction"
git add ".github\workflows\diabetes-ann-ci.yml"
git commit -m "Update diabetes prediction project"
git pull --rebase origin main
git push origin main
git status
```

Streamlit Community Cloud monitors the connected `main` branch. A successful push normally triggers an automatic rebuild or app restart.

## 3. Deployment configuration reference

When recreating the deployment, use:

```text
Repository: unit-mole/ann-deep-learning-projects
Branch: main
Main file path: 05-diabetes-prediction/app/streamlit_app.py
Python version: 3.12
```

The current public application URL is:

```text
https://ann-deep-learning-projects-bczyq9q5aa8eqbvqskqyar.streamlit.app/
```

No values are required in Streamlit **Secrets**.

## 4. Monorepo dependency behavior

Community Cloud searches the entrypoint directory before the repository root. Because the app entrypoint is:

```text
05-diabetes-prediction/app/streamlit_app.py
```

an app-specific dependency file is maintained at:

```text
05-diabetes-prediction/app/requirements.txt
```

The project-level `requirements.txt` is used for local development and CI. Keep the two runtime dependency files synchronized. The development file `requirements-dev.txt` additionally includes testing and notebook-validation packages.

## 5. Post-deployment verification

After each material update:

1. Open the live URL in a normal browser window.
2. Open it again in an Incognito/InPrivate window.
3. Test one manual prediction.
4. Run the included batch sample.
5. Upload a valid CSV and confirm scoring.
6. Upload an invalid CSV and confirm that a clear validation message appears.
7. Confirm that the model-card metrics and feature-importance chart load.
8. Check the GitHub Actions workflow for a green status.

## 6. Common troubleshooting

### Model artifact not found

Confirm that these files are visible on GitHub:

```text
models/diabetes_ann.keras
models/preprocessor.joblib
models/model_metadata.json
```

The app uses project-root-relative paths and does not rely on local Windows paths.

### TensorFlow or Keras installation failure

Confirm that Streamlit is using Python `3.12` and that both dependency files include the pinned versions:

```text
tensorflow-cpu==2.20.0
keras==3.15.0
```

Changing the Python version may require rebooting or recreating the Streamlit deployment.

### Uploaded CSV cannot be scored

Use the schema in `data/sample_input.csv`. The required columns are:

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
```

Column spelling and capitalization must match. Empty files, missing columns, unexpected text, infinite values, and negative measurements are rejected with a user-facing validation message.

### App does not update after a GitHub push

Open the app-management page in Streamlit Community Cloud and select **Reboot app**. Review the build logs if the new commit is not reflected.

### Memory or cold-start concerns

The trained model is small and is loaded with `st.cache_resource`, so the artifacts are loaded once per application process rather than on every interaction.

## Optional Hugging Face Spaces route

Hugging Face Spaces can also host a Streamlit-compatible interface, but Streamlit Community Cloud remains the preferred deployment for this Streamlit-native monorepo. A Space would be useful only if the project is later paired with additional Hugging Face model-card assets.
