# Hosting Guide

## Recommended option: Streamlit Community Cloud

The project is already a Streamlit application, the trained model artifacts are
small enough to keep in Git, and Community Cloud can deploy directly from the
public monorepo.

The locally validated and deployed environment uses **Python 3.12**.

## Required repository files

- `app/streamlit_app.py`
- `app/requirements.txt`
- `models/final_credit_risk_ann_model.keras`
- `models/preprocessing_schema.json`
- `models/project_metadata.json`
- `src/`
- `data/sample_input.csv`

## Current deployment

- **Repository:** `unit-mole/ann-deep-learning-projects`
- **Branch:** `main`
- **Entrypoint:** `03-credit-risk-probability-scoring/app/streamlit_app.py`
- **Python:** `3.12`
- **Live application:**  
  https://ann-deep-learning-projects-9p9vupmu9kbk5462v6hbkb.streamlit.app/

## Initial deployment steps

1. Push the project and workflow files to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Select **Create app** and choose the repository and `main` branch.
4. Set the main file path to:

   ```text
   03-credit-risk-probability-scoring/app/streamlit_app.py
   ```

5. Open **Advanced settings** and select Python 3.12.
6. Leave Secrets empty because this application does not use credentials.
7. Deploy and monitor the build logs.
8. Test manual scoring, sample scoring, CSV upload, the risk chart, and CSV
   download.
9. Verify the public link in an Incognito or InPrivate browser.
10. Add the live URL to the project README, main repository roadmap, resume,
    LinkedIn Featured section, and portfolio website.

## Dependency files in the monorepo

Community Cloud resolves dependencies most reliably when `requirements.txt`
sits beside the Streamlit entrypoint. Keep these two files synchronized:

```text
03-credit-risk-probability-scoring/requirements.txt
03-credit-risk-probability-scoring/app/requirements.txt
```

On Windows Command Prompt:

```bat
copy /Y "03-credit-risk-probability-scoring\requirements.txt" "03-credit-risk-probability-scoring\app\requirements.txt"
```

Confirm they are identical:

```bat
fc "03-credit-risk-probability-scoring\requirements.txt" "03-credit-risk-probability-scoring\app\requirements.txt"
```

## Reproducible environments

`requirements.txt` contains supported dependency ranges. After validating a
local environment, create an exact snapshot:

```bat
python -m pip freeze > requirements-lock.txt
```

Recreate that exact environment with:

```bat
python -m pip install -r requirements-lock.txt
```

Regenerate the lock file only after retesting the application, model loading,
sample scoring, unit tests, and Streamlit deployment.

## Deployment verification checklist

After every dependency or model-artifact change, confirm:

```text
[ ] Application starts without an exception
[ ] Final ANN artifact loads
[ ] Preprocessing schema loads
[ ] Sample data produces Low, Medium, and High Risk results
[ ] Manual low-risk scoring works
[ ] Manual high-risk scoring works
[ ] CSV upload and batch scoring work
[ ] Risk distribution chart renders
[ ] Downloaded scored CSV opens correctly
[ ] Public URL works without authentication
```

## Common fixes

### Module not found

Confirm:

- `app/requirements.txt` is committed.
- The entrypoint is exactly
  `03-credit-risk-probability-scoring/app/streamlit_app.py`.
- Imports are resolved from the project root by `app/streamlit_app.py`.

### Model artifact not found

Confirm these files are tracked:

```text
models/final_credit_risk_ann_model.keras
models/preprocessing_schema.json
models/project_metadata.json
```

### Build or memory issue

- Keep development-only packages out of `requirements.txt`.
- Place pytest, Ruff, and notebook utilities in `requirements-dev.txt`.
- Remove unused packages only after testing locally.
- Review Streamlit build logs before changing model code.

### Python incompatibility

The reference deployment uses Python 3.12. If the app was created with another
version, delete and redeploy it with Python 3.12 selected.

### Dependency drift

Use the locally tested `requirements-lock.txt` to identify which installed
version changed. Update the supported ranges only after verifying the project
locally and in Streamlit Community Cloud.

## Updating the deployed application

Community Cloud redeploys from the linked GitHub branch.

Recommended update sequence:

```bat
python -m pytest -q
python -m compileall app src tests
git status
git add <changed-files>
git commit -m "Describe the deployment change"
git pull --rebase origin main
git push origin main
```

Then open the Streamlit logs and verify that the application restarts
successfully.

## Optional model-artifact smoke workflow

The repository can include:

```text
.github/workflows/credit-risk-model-smoke.yml
```

This manually triggered and weekly workflow installs the deployment
dependencies and verifies that the ANN and preprocessing artifacts load
successfully. It is intentionally separate from the lightweight push CI because
TensorFlow installation is slower.

## Alternative: Hugging Face Spaces

Hugging Face no longer treats Streamlit as a default built-in Space SDK, so use
the included Dockerfile:

1. Create a new **Docker Space**.
2. Copy the project files to the Space repository.
3. Keep port `8501`.
4. Push and review the Docker build logs.
5. Test the same manual, sample, batch, chart, and download workflows.

This option is useful when the project should also appear in an ML-focused
public profile, while Streamlit Community Cloud remains the simpler primary
deployment.

## Local Docker test

```bash
docker build -t credit-risk-ann .
docker run --rm -p 8501:8501 credit-risk-ann
```

Then open:

```text
http://localhost:8501
```
