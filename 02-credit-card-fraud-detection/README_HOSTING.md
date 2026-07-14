# Hosting Guide

## Recommended Platform: Streamlit Community Cloud

Streamlit Community Cloud is the preferred hosting option because this project
is already built with Streamlit and deploys directly from GitHub without a
Dockerfile.

### Current deployment

- **GitHub repository:** `unit-mole/ann-deep-learning-projects`
- **Branch:** `main`
- **Application entrypoint:** `02-credit-card-fraud-detection/app/streamlit_app.py`
- **Python version:** `3.12`
- **Live application:**  
  https://ann-deep-learning-projects-dqnj5rpwbpmuxtd2tcm5mh.streamlit.app/

---

## Files Required for Deployment

```text
ann-deep-learning-projects/
├── .github/
│   └── workflows/
│       └── credit-card-fraud-ci.yml
└── 02-credit-card-fraud-detection/
    ├── app/
    │   ├── requirements.txt
    │   └── streamlit_app.py
    ├── data/
    │   ├── README_data.md
    │   └── sample_input.csv
    ├── models/
    │   ├── credit_card_fraud_ann.keras
    │   ├── credit_card_scaler.pkl
    │   ├── decision_threshold.json
    │   └── feature_schema.json
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   └── prediction_pipeline.py
    ├── README.md
    ├── README_HOSTING.md
    ├── requirements.txt
    └── requirements-dev.txt
```

The full source dataset is not required for inference and must remain excluded:

```text
02-credit-card-fraud-detection/data/creditcard.csv
```

The small demonstration file must remain committed:

```text
02-credit-card-fraud-detection/data/sample_input.csv
```

## Why `app/requirements.txt` Exists

The Streamlit entrypoint is inside a project subfolder:

```text
02-credit-card-fraud-detection/app/streamlit_app.py
```

A complete dependency file is kept beside the entrypoint so Streamlit Community
Cloud can install the required packages reliably within the monorepo.

Keep `app/requirements.txt` synchronized with the project-level
`requirements.txt`.

---

## Step 1: Test Locally

```bat
cd /d "C:\Users\atripathi\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\ann-deep-learning-projects\02-credit-card-fraud-detection"
```

Create and activate a Python 3.12 environment when needed:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Install dependencies and run checks:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m compileall app src tests
```

Launch the app:

```bat
python -m streamlit run app\streamlit_app.py
```

Verify the demo sample, CSV upload, manual scoring, threshold control, and CSV
download. Stop the app with `Ctrl + C`.

---

## Step 2: Publish Updates to GitHub

This project belongs inside the existing monorepo. Do not create a separate
repository.

```bat
cd /d "C:\Users\atripathi\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\ann-deep-learning-projects"
git status
git add "02-credit-card-fraud-detection"
git add ".github\workflows\credit-card-fraud-ci.yml"
git commit -m "Update Credit Card Fraud Detection project"
git push origin main
```

Before pushing, confirm these are not staged:

```text
02-credit-card-fraud-detection/data/creditcard.csv
02-credit-card-fraud-detection/.venv/
02-credit-card-fraud-detection/.pytest_cache/
```

---

## Step 3: Verify GitHub Actions

Open the repository **Actions** tab and select:

```text
Credit Card Fraud Project CI
```

Confirm green checks for checkout, Python setup, dependency installation,
compilation, and unit tests.

---

## Step 4: Streamlit Deployment Coordinates

```text
Repository:
unit-mole/ann-deep-learning-projects

Branch:
main

Main file path:
02-credit-card-fraud-detection/app/streamlit_app.py

Python:
3.12
```

No secrets are required.

Current live URL:

```text
https://ann-deep-learning-projects-dqnj5rpwbpmuxtd2tcm5mh.streamlit.app/
```

Commits pushed to `main` are automatically picked up by Streamlit Community
Cloud.

---

## Step 5: Test the Public App

Open the live URL in an incognito/private window and confirm it does not require
an invitation.

Test:

1. Use demo sample
2. Upload CSV
3. Manual single transaction
4. Threshold adjustment
5. Prediction CSV download

For the supplied reference model, the demonstration sample should show:

```text
Default threshold: 0.50
Transactions scored: 50
SAFE: 43
FRAUD / RISK: 7
Predicted fraud rate: 14.00%
```

---

## Troubleshooting

### Dependency installation fails

Confirm:

- `app/requirements.txt` exists;
- it contains the complete dependency list;
- Python 3.12 is selected;
- versions match the tested local environment.

### Model file not found

Confirm these files are committed:

```text
models/credit_card_fraud_ann.keras
models/credit_card_scaler.pkl
models/feature_schema.json
models/decision_threshold.json
```

### Demo sample not found

Confirm:

```text
data/sample_input.csv
```

### TensorFlow startup is slow

The first deployment or restart may take several minutes. The hosted app loads
saved artifacts for inference and does not retrain the model.

### Git push is rejected with `fetch first`

```bat
git status
git pull --rebase origin main
git push origin main
```

Preserve or commit local edits before pulling. Do not force-push unless
intentionally replacing remote history.

---

## Live Links to Maintain

Project README:

```markdown
**Live demo:** [Open the Streamlit application](https://ann-deep-learning-projects-dqnj5rpwbpmuxtd2tcm5mh.streamlit.app/)
```

Root portfolio README:

```markdown
| 2 | [Credit Card Fraud Detection](02-credit-card-fraud-detection/) | Imbalanced binary classification | [Live Demo](https://ann-deep-learning-projects-dqnj5rpwbpmuxtd2tcm5mh.streamlit.app/) |
```

Suggested repository topics:

```text
artificial-neural-network
credit-card-fraud
fraud-detection
imbalanced-classification
tensorflow
keras
streamlit
machine-learning
data-science
risk-analytics
```

---

## Sharing the Project

### Resume

```text
Credit Card Fraud Detection | TensorFlow, Keras, scikit-learn, Streamlit
Built an ANN fraud-risk scoring pipeline for a 0.17%-fraud dataset using
imbalance-aware evaluation, probability scoring, threshold controls, automated
tests, and a deployed batch-prediction application.
```

### LinkedIn

Share the live URL, GitHub link, a clear app screenshot, fraud-specific metrics,
and the reason accuracy alone is misleading for rare-event detection.

### Portfolio Website

Use two buttons:

```text
View Source Code
Open Live Demo
```
