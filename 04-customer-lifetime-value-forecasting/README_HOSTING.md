# Hosting Guide — Customer Lifetime Value Forecasting

## Current Deployment Status

**Status:** Successfully deployed  
**Platform:** Streamlit Community Cloud  
**Repository:** `unit-mole/ann-deep-learning-projects`  
**Branch:** `main`  
**Python version:** `3.12`  
**Application entrypoint:**

```text
04-customer-lifetime-value-forecasting/app/streamlit_app.py
```

**Live application:**  
[Open the deployed Streamlit application](https://ann-deep-learning-projects-u4gymvvpwuaowqnmkjq3wa.streamlit.app/)

The GitHub Actions workflow is installed at:

```text
.github/workflows/clv-ann-ci.yml
```

The workflow runs automatically when relevant CLV project files or the workflow
configuration are changed.

---

## Recommended Platform

Streamlit Community Cloud is the selected hosting platform for this project.

It is a strong fit because:

- The application is written in Streamlit.
- It connects directly to the GitHub repository and `main` branch.
- Relevant commits automatically trigger application updates.
- No Dockerfile or separate server configuration is required.
- The public URL is suitable for GitHub, LinkedIn, a resume, and a portfolio.

Hugging Face Spaces remains an alternative, but a Docker-based setup introduces
additional configuration that is unnecessary for this portfolio application.

---

## Deployment Files

Keep these files committed to the repository:

```text
04-customer-lifetime-value-forecasting/
├── app/
│   ├── streamlit_app.py
│   └── requirements.txt
├── data/
│   └── sample_input.csv
├── models/
│   ├── clv_ann_model.keras
│   ├── label_encoders.pkl
│   ├── numeric_scaler.pkl
│   └── model_metadata.json
├── outputs/
│   ├── model_metrics.json
│   └── *.png
└── src/
    └── *.py
```

The trained model is approximately 1.1 MB, so standard Git is sufficient and
Git LFS is not required.

---

## Current Streamlit Configuration

| Setting | Value |
|---|---|
| Repository | `unit-mole/ann-deep-learning-projects` |
| Branch | `main` |
| Main file path | `04-customer-lifetime-value-forecasting/app/streamlit_app.py` |
| Python version | `3.12` |
| Dependency file | `04-customer-lifetime-value-forecasting/app/requirements.txt` |
| Secrets | None required |
| Live URL | `https://ann-deep-learning-projects-u4gymvvpwuaowqnmkjq3wa.streamlit.app/` |

---

## Test Locally

Clone the repository and enter the CLV project folder:

```bat
git clone https://github.com/unit-mole/ann-deep-learning-projects.git

cd ann-deep-learning-projects\04-customer-lifetime-value-forecasting
```

Create and activate the virtual environment:

```bat
py -3.12 -m venv .venv

.venv\Scripts\activate.bat
```

Install the dependencies:

```bat
python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt -r requirements-ci.txt
```

Run the automated tests:

```bat
python -m pytest -q
```

Start the application:

```bat
python -m streamlit run app\streamlit_app.py
```

Verify all four application sections:

1. Overview
2. Single Customer
3. Batch Scoring
4. Model Performance

Also test `data/sample_input.csv` and confirm that the scored CSV downloads
successfully.

---

## GitHub and CI Configuration

The project is stored at:

```text
ann-deep-learning-projects/
└── 04-customer-lifetime-value-forecasting/
```

The CI workflow is already installed at:

```text
.github/workflows/clv-ann-ci.yml
```

GitHub Actions validates the project automatically when relevant files are
pushed. The workflow should remain at the repository level under
`.github/workflows/`; it should not be moved back into the project folder.

For future updates, run these commands from the repository root:

```bat
git status

git add 04-customer-lifetime-value-forecasting .github\workflows\clv-ann-ci.yml

git commit -m "Update Customer Lifetime Value Forecasting project"

git pull --rebase origin main

git push origin main
```

---

## Updating the Live Application

The Streamlit application is connected to the `main` branch.

To publish an update:

1. Test the change locally.
2. Commit and push the relevant files to `main`.
3. Confirm that the CLV GitHub Actions workflow passes.
4. Allow Streamlit Community Cloud to rebuild or restart the application.
5. Open the live URL and repeat the application validation checklist.

**Live URL:**  
[https://ann-deep-learning-projects-u4gymvvpwuaowqnmkjq3wa.streamlit.app/](https://ann-deep-learning-projects-u4gymvvpwuaowqnmkjq3wa.streamlit.app/)

---

## Live Application Validation Checklist

After each deployment update, confirm that:

- The application opens without an error.
- The Overview section displays the project metrics.
- Single-customer forecasting returns CLV, retention probability, segment, and recommendation.
- Batch scoring works with the included sample file.
- CSV upload works.
- The scored CSV downloads successfully.
- All model-performance charts load.
- The application also works in a private or incognito browser window.

---

## Build Troubleshooting

### TensorFlow or Keras model-load error

Confirm that Streamlit Community Cloud is using Python 3.12 and that these files
are committed:

```text
models/clv_ann_model.keras
models/model_metadata.json
```

The saved artifact reports Keras 3.14.0, which is pinned in the requirements
files.

### `ModuleNotFoundError: src`

Use the provided `app/streamlit_app.py` unchanged. It adds the project root to
`sys.path` before importing project modules.

### File not found

Do not use local Windows absolute paths inside the application. The application
uses `pathlib` and repository-relative paths so it works on both Windows and
Linux.

### Slow first launch

TensorFlow can create a heavier cold start than a simple analytics application.
Subsequent reruns should use Streamlit's cached model resource.

### Dependency changes not reflected

Commit and push the updated dependency file. Streamlit Community Cloud detects
dependency changes and rebuilds the environment.

### Application does not refresh after a push

Open the application settings in Streamlit Community Cloud, review the build
logs, and restart or reboot the application if required. Confirm that the
deployment still points to the `main` branch and the documented entrypoint.

---

## Redeployment Configuration

If the application ever needs to be recreated in Streamlit Community Cloud, use:

```text
Repository: unit-mole/ann-deep-learning-projects
Branch: main
Main file path: 04-customer-lifetime-value-forecasting/app/streamlit_app.py
Python version: 3.12
Secrets: None
```

After redeployment, test the live application before updating any public links.

---

## Alternative — Hugging Face Spaces

Use Hugging Face Spaces only when the project specifically needs to appear on a
Hugging Face profile. A Docker Space can install the same dependencies, expose
port 8501, and start the application with:

```text
streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

For this project, Streamlit Community Cloud remains the simpler and more
recruiter-friendly hosting option.
