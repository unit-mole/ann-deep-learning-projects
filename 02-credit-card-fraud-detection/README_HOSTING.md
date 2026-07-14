# Hosting Guide

## Recommended Option: Streamlit Community Cloud

Streamlit Community Cloud is the best fit for this project because:

- the application is already built with Streamlit;
- it deploys directly from GitHub;
- no Dockerfile is required;
- commits pushed to GitHub are reflected in the deployed application;
- the final public link uses a shareable `streamlit.app` subdomain;
- it is simpler than Hugging Face Spaces for a Streamlit application.

Hugging Face Spaces remains a valid option, but Streamlit is no longer offered
as a default built-in Spaces SDK. A new Streamlit deployment on Hugging Face
generally requires a Docker Space, which adds a Dockerfile and more deployment
configuration.

Official references:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
- https://huggingface.co/docs/hub/spaces-changelog
- https://huggingface.co/docs/hub/spaces-sdks-docker

---

## Files Required for Deployment

The following files must be committed:

```text
ANN/
├── .streamlit/
│   └── config.toml
└── Credit_Card_Fraud_Detection/
    ├── app/
    │   ├── requirements.txt
    │   └── streamlit_app.py
    ├── data/
    │   └── sample_input.csv
    ├── models/
    │   ├── credit_card_fraud_ann.keras
    │   ├── credit_card_scaler.pkl
    │   ├── feature_schema.json
    │   └── decision_threshold.json
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   └── prediction_pipeline.py
    └── requirements.txt
```

The full `data/creditcard.csv` file is not required for inference and should not
be pushed to GitHub.

### Why `app/requirements.txt` Exists

In the planned ANN monorepo, the Streamlit entrypoint is:

```text
Credit_Card_Fraud_Detection/app/streamlit_app.py
```

Community Cloud looks for a dependency file in the entrypoint directory and
then at the repository root. The file:

```text
Credit_Card_Fraud_Detection/app/requirements.txt
```

contains:

```text
-r ../requirements.txt
```

This allows the app to use the project-level dependencies while keeping a clean
project structure.

---

## Step 1: Test Locally

From the project folder:

```bash
cd ANN/Credit_Card_Fraud_Detection
```

Create and activate a Python 3.11 virtual environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch the app:

```bash
streamlit run app/streamlit_app.py
```

Verify all three modes:

1. Use demo sample.
2. Upload `data/sample_input.csv`.
3. Enter a manual row.

Also verify:

- the model loads without errors;
- the sample preview appears;
- probabilities and labels are produced;
- threshold changes affect the number of fraud alerts;
- the CSV download works.

---

## Step 2: Create the GitHub Repository

Recommended repository name:

```text
ANN
```

Recommended visibility:

```text
Public
```

A public repository is easiest to present to recruiters and deploy as a public
portfolio application. Do not commit credentials, API keys, private data, or
customer information.

From the folder containing `ANN`:

```bash
cd ANN
git init
git branch -M main
git add .
git commit -m "Add ANN credit card fraud detection portfolio project"
git remote add origin https://github.com/<your-github-username>/ANN.git
git push -u origin main
```

Before pushing, confirm the full dataset is excluded:

```bash
git status
```

You should not see:

```text
Credit_Card_Fraud_Detection/data/creditcard.csv
```

You should see the sample file and model artifacts.

---

## Step 3: Connect GitHub to Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud using your GitHub account.
2. Open the Community Cloud workspace.
3. Choose **Create app**.
4. Select the GitHub repository:
   ```text
   <your-github-username>/ANN
   ```
5. Select the branch:
   ```text
   main
   ```
6. Enter the app file path:
   ```text
   Credit_Card_Fraud_Detection/app/streamlit_app.py
   ```
7. Open **Advanced settings**.
8. Select **Python 3.11** for consistent TensorFlow compatibility.
9. Choose an available app URL, for example:
   ```text
   anmol-credit-card-fraud-ann
   ```
10. Click **Deploy**.

A possible final URL will look like:

```text
https://anmol-credit-card-fraud-ann.streamlit.app
```

The exact URL depends on the subdomain you select and its availability.

---

## Step 4: Test the Deployed App

After the build completes:

1. Open the public URL in a private/incognito browser window.
2. Use the preloaded demo sample.
3. Upload `sample_input.csv`.
4. Change the decision threshold.
5. Download the predictions.
6. Confirm the app works without requiring GitHub sign-in.
7. Open **Manage app** and review logs if an error occurs.

Common deployment issues:

### Dependency installation fails

Confirm:

- `app/requirements.txt` exists;
- it contains `-r ../requirements.txt`;
- only one active dependency chain is used;
- Python 3.11 was selected;
- package names are spelled correctly.

### Model file not found

Confirm these files were committed:

```text
models/credit_card_fraud_ann.keras
models/credit_card_scaler.pkl
models/feature_schema.json
models/decision_threshold.json
```

The application resolves paths relative to the project, not the terminal
working directory.

### TensorFlow build is slow or memory-constrained

- keep the full dataset out of the deployment;
- do not train the model inside the Streamlit application;
- use the small saved inference model;
- restart the app after a dependency or artifact change.

### Pickle compatibility warning

The project pins scikit-learn to the version used for the audited package.
Retrain and resave the scaler if you intentionally change major scikit-learn
versions.

---

## Step 5: Add the Live Link to GitHub

Replace the placeholder in `README.md`:

```text
https://<your-custom-subdomain>.streamlit.app
```

with the deployed link.

Add a Streamlit badge:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-custom-subdomain>.streamlit.app)
```

Also add the URL to the GitHub repository **About** section.

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

## Step 6: Share the Project

### Resume

```text
Credit Card Fraud Detection | TensorFlow, Keras, scikit-learn, Streamlit
Built an ANN fraud-risk scoring pipeline for a 0.17%-fraud dataset, using
imbalance-aware evaluation, probability scoring, threshold controls, and a
deployed batch-prediction application.
```

### LinkedIn

Share:

- a clear app screenshot;
- the live Streamlit URL;
- the GitHub repository;
- the business reason accuracy is misleading;
- the precision, recall, F1, ROC-AUC, and PR-AUC results;
- the technical improvements made for deployment.

### Portfolio Website

Include two buttons:

```text
View Source Code
Open Live Demo
```

---

## Optional: Hugging Face Spaces

Use Hugging Face only when you specifically want the application displayed on
your Hugging Face profile. For a new Streamlit app, create a **Docker Space**
and provide:

- a Dockerfile;
- the application code;
- model artifacts;
- requirements;
- port `7860`;
- `sdk: docker` and `app_port: 7860` in the Space metadata.

For this project, the Docker route adds complexity without improving the
portfolio demo, so Streamlit Community Cloud remains the recommended option.
