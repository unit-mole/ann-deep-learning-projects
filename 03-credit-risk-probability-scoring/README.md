# Credit Risk Probability Scoring using Artificial Neural Networks

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://ann-deep-learning-projects-9p9vupmu9kbk5462v6hbkb.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Credit Risk ANN CI](https://github.com/unit-mole/ann-deep-learning-projects/actions/workflows/credit-risk-ann-ci.yml/badge.svg)](https://github.com/unit-mole/ann-deep-learning-projects/actions/workflows/credit-risk-ann-ci.yml)

An end-to-end credit-risk probability scoring project that uses an Artificial Neural Network
to estimate applicant default risk and assign operational risk categories. The repository
includes reproducible preprocessing and training code, imbalance-aware evaluation,
probability-based scoring, threshold controls, saved model artifacts, risk recommendations,
and a Streamlit application for manual and batch scoring.

**Status:** Portfolio-ready  
**Live demo:** [Open the Streamlit application](https://ann-deep-learning-projects-9p9vupmu9kbk5462v6hbkb.streamlit.app/)  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ann-deep-learning-projects-9p9vupmu9kbk5462v6hbkb.streamlit.app/)  
**Primary stack:** Python · Keras · TensorFlow · scikit-learn · Streamlit

---

## Business Problem

Lenders need more than a binary approve/reject prediction. They need a comparable
risk score that can rank applicants, support manual-review queues, and make
decision policies transparent. This project answers:

> Given applicant information, what is the probability that the applicant
> belongs to a higher default-risk category?

The deployed pipeline returns:

- **Risk probability score**
- **Low / Medium / High risk category**
- **Decision recommendation and review priority**

## Project Objective

Build a portfolio-ready ANN solution that can:

1. Validate and preprocess applicant-level credit data.
2. Engineer affordability and collateral-related features.
3. Learn non-linear credit-risk patterns using a deep neural network.
4. Handle class imbalance using training-set class weights.
5. Produce probability-of-default scores rather than class labels alone.
6. Convert probabilities into operational risk categories and review actions.
7. Support individual, sample, and batch scoring in an interactive application.
8. Save and reload the artifacts required for reproducible inference.

## Portfolio Scope

This is an educational demonstration built on a deterministic **synthetic banking
dataset**. It is not a production credit model and must not be used for real
lending decisions.

## Dataset

The notebook generates 25,000 applicant records with demographic, financial,
credit-history, collateral, and loan-purpose fields. The observed target
distribution is:

| Class | Records | Share |
|---|---:|---:|
| Non-default | 19,682 | 78.73% |
| Default | 5,318 | 21.27% |

Six fields receive 1% controlled missingness to demonstrate imputation. No
private customer data is included in GitHub.

## Tools and Technologies

| Area | Technology |
|---|---|
| Language | Python |
| Data processing | pandas, NumPy |
| Modeling | TensorFlow / Keras |
| Preprocessing | scikit-learn |
| Evaluation | scikit-learn, Matplotlib |
| Interactive charts | Plotly |
| Demo application | Streamlit |
| Model persistence | Keras `.keras`, Joblib, JSON |
| Testing / quality | pytest, compile checks, GitHub Actions |
| Hosting | Streamlit Community Cloud |

## Project Workflow

```text
Applicant-level data
        │
        ▼
Schema validation and controlled missingness
        │
        ▼
Feature engineering
        │
        ▼
Stratified 70% / 15% / 15% split
        │
        ▼
Numerical imputation and scaling
        │
        ▼
Categorical imputation and one-hot encoding
        │
        ▼
ANN training with balanced class weights
        │
        ▼
Validation threshold selection
        │
        ▼
Held-out test evaluation and calibration checks
        │
        ▼
Saved model + preprocessing schema + metadata
        │
        ▼
Streamlit manual, sample, and batch scoring
```

## Feature Engineering

| Derived field | Formula | Purpose |
|---|---|---|
| Monthly income | Annual income / 12 | Monthly affordability proxy |
| Loan-to-income | Loan amount / annual income | Relative debt burden |
| Collateral ratio | Collateral value / loan amount | Security coverage proxy |

The portable preprocessing schema stores training medians, scaling statistics,
categorical modes, category order, and final feature order. This makes the demo
less dependent on scikit-learn pickle compatibility.

## ANN Architecture

```text
41 processed inputs
      ↓
Dense 256 + ReLU + L2 + BatchNorm + Dropout(0.30)
      ↓
Dense 128 + ReLU + L2 + BatchNorm + Dropout(0.25)
      ↓
Dense 64 + ReLU + L2 + BatchNorm + Dropout(0.20)
      ↓
Dense 32 + ReLU + Dropout(0.10)
      ↓
Sigmoid probability of default
```

Training uses Adam, binary cross-entropy, ROC-AUC, PR-AUC, balanced class
weights, early stopping, and `ReduceLROnPlateau`.

## Probability and Decision Logic

The ANN outputs a continuous probability between 0 and 1. Two threshold layers
are intentionally separated:

- **Classification threshold: 0.58** — selected on the validation set to
  maximize F1.
- **Operational risk bands:**
  - `< 0.20`: Low Risk
  - `0.20–<0.50`: Medium Risk
  - `>= 0.50`: High Risk

Risk bands are a policy layer for demonstration. A real lender would set them
using expected-loss economics, approval capacity, portfolio appetite,
calibration, fairness review, and regulation—not validation F1 alone.

## Model Results

| Metric | Test result |
|---|---:|
| Accuracy | 0.774 |
| Precision | 0.477 |
| Recall | 0.649 |
| F1-score | 0.550 |
| ROC-AUC | 0.813 |
| PR-AUC | 0.610 |
| Log loss | 0.518 |
| Brier score | 0.174 |

Confusion matrix at threshold 0.58:

| | Predicted non-default | Predicted default |
|---|---:|---:|
| **Actual non-default** | 2,384 | 568 |
| **Actual default** | 280 | 518 |

A **false negative** is a risky applicant treated as lower risk and can create
credit loss. A **false positive** is a safer applicant flagged as risky and can
reduce approvals, revenue, and customer access. Threshold selection therefore
requires business-cost analysis, not accuracy alone.

## Class Imbalance

Defaults are only 21.27% of the synthetic data. The notebook uses balanced class
weights (`0: 0.635`, `1: 2.351`) rather than generating synthetic minority
examples. This retains original training rows while making default errors more
influential. Precision-recall analysis and validation threshold tuning
complement the class weighting.

## Explainability

Permutation importance identifies the strongest model-performance drivers in
the executed notebook. The leading signals include loan-to-income ratio,
debt-to-income ratio, monthly income, delinquency count, revolving utilization,
grade, home ownership, and collateral value.

These are global associations—not causal explanations or legally sufficient
adverse-action reasons.

![Permutation importance](outputs/permutation_importance.png)

## Visual Model Results

| Confusion matrix | ROC curve |
|---|---|
| ![Confusion matrix](outputs/confusion_matrix.png) | ![ROC curve](outputs/roc_curve.png) |

| Precision-recall curve | Calibration curve |
|---|---|
| ![PR curve](outputs/precision_recall_curve.png) | ![Calibration curve](outputs/calibration_curve.png) |

| Training AUC | Risk bands |
|---|---|
| ![Training AUC](outputs/training_auc.png) | ![Risk category distribution](outputs/risk_category_distribution.png) |

## Streamlit Demo

The application supports:

- Manual applicant entry
- CSV upload for batch scoring
- Preloaded safe sample data
- Probability, category, predicted class, and recommendation
- Risk-category distribution chart
- Downloadable scored CSV
- Input-schema guidance for reviewers

### Application Overview

The main demonstration view connects model probabilities to business-facing
risk categories, recommendations, and review priorities.

![Credit Risk Probability Scoring Streamlit demo](images/demo_screenshot.png)

### Manual Risk Examples

The manual-scoring workflow demonstrates how applicant characteristics affect
the risk output and resulting recommendation.

<p align="center">
  <img src="images/06_manual_low_risk_prediction.png" width="48%" alt="Manual low-risk applicant prediction">
  <img src="images/07_manual_high_risk_prediction.png" width="48%" alt="Manual high-risk applicant prediction">
</p>

### CSV Batch Scoring

The batch workflow previews applicant data, scores all records, summarizes the
risk distribution, and provides a downloadable scored CSV.

![Batch scoring results](images/09_batch_scoring_results.png)

<details>
<summary><strong>View additional application screenshots</strong></summary>

### Application Home

![Application home](images/01_app_home.png)

### Sample Input Preview

![Sample input preview](images/02_sample_input_preview.png)

### Sample Scoring Summary

![Sample scoring summary](images/03_sample_scoring_summary.png)

### Risk Category Distribution

![Risk category distribution](images/04_risk_category_distribution.png)

### Detailed Scored Table

![Detailed scored table](images/05_sample_scored_table.png)

### Batch Upload Preview

![Batch upload preview](images/08_batch_upload_preview.png)

### Input Schema and Interpretation

![Input schema and interpretation](images/10_input_schema_information.png)

</details>

## Model Artifacts

| Artifact | Purpose |
|---|---|
| `models/final_credit_risk_ann_model.keras` | Primary ANN artifact used by the deployed application |
| `models/best_credit_risk_ann_model.keras` | Best validation checkpoint retained for comparison and reproducibility |
| `models/preprocessor.joblib` | Serialized preprocessing pipeline |
| `models/preprocessing_schema.json` | Portable medians, scales, category order, and feature order |
| `models/project_metadata.json` | Threshold, feature lists, seed, and input dimension |

To verify whether the two `.keras` files are byte-for-byte identical on Windows:

```bat
certutil -hashfile "models\best_credit_risk_ann_model.keras" SHA256
certutil -hashfile "models\final_credit_risk_ann_model.keras" SHA256
```

Matching hashes indicate that the best checkpoint and final saved model are the
same artifact.

## Run Locally

### 1. Open the project directory

```bash
cd ann-deep-learning-projects/03-credit-risk-probability-scoring
```

### 2. Create and activate a virtual environment

Windows Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

For the supported dependency ranges:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the exact locally tested environment, generate a lock file after successful
installation:

```bash
python -m pip freeze > requirements-lock.txt
```

Install development tools when needed:

```bash
python -m pip install -r requirements-dev.txt
```

### 4. Run tests

```bash
python -m pytest -q
python -m compileall src app
```

### 5. Launch the supplied pretrained demo

The trained ANN models, preprocessing artifacts, metadata, and sample CSV are
already included.

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local address displayed in the terminal, normally:

```text
http://localhost:8501
```

### 6. Optional: retrain the model

To reproduce training on the deterministic synthetic dataset:

```bash
python -m src.model_training
```

To train on a compatible CSV containing `default_flag`:

```bash
python -m src.model_training --data path/to/training_data.csv
```

Training writes model checkpoints to `models/` and logs and evaluation outputs
to `outputs/`.

## Deploy

The application is deployed through Streamlit Community Cloud directly from
the public ANN portfolio repository.

- **Repository:** `unit-mole/ann-deep-learning-projects`
- **Branch:** `main`
- **Entrypoint:** `03-credit-risk-probability-scoring/app/streamlit_app.py`
- **Python:** `3.12`
- **Live application:**  
  https://ann-deep-learning-projects-9p9vupmu9kbk5462v6hbkb.streamlit.app/

The `app/requirements.txt` file contains the deployment dependency list beside
the Streamlit entrypoint. This allows Community Cloud to resolve the
environment reliably within the monorepo.

See [`README_HOSTING.md`](README_HOSTING.md) for deployment and maintenance
instructions.

## Project Structure

```text
ann-deep-learning-projects/
├── .github/
│   └── workflows/
│       ├── credit-risk-ann-ci.yml
│       └── credit-risk-model-smoke.yml
├── 01-churn-classification/
├── 02-credit-card-fraud-detection/
├── 03-credit-risk-probability-scoring/
│   ├── app/
│   │   ├── requirements.txt
│   │   └── streamlit_app.py
│   ├── data/
│   │   ├── README_data.md
│   │   └── sample_input.csv
│   ├── images/
│   ├── models/
│   ├── notebooks/
│   ├── outputs/
│   ├── src/
│   ├── tests/
│   ├── .gitignore
│   ├── .streamlit/
│   │   └── config.toml
│   ├── Dockerfile
│   ├── LICENSE
│   ├── MODEL_CARD.md
│   ├── PROJECT_FILE_INVENTORY.json
│   ├── README.md
│   ├── README_HOSTING.md
│   ├── requirements-dev.txt
│   ├── requirements-lock.txt
│   └── requirements.txt
├── .gitignore
├── LICENSE
├── PROJECT_ROADMAP.md
└── README.md
```

## Testing and CI

Run lightweight unit tests:

```bash
python -m pytest -q
```

Check Python files for syntax errors:

```bash
python -m compileall app src tests
```

The standard CI workflow runs on relevant pushes and pull requests:

```text
.github/workflows/credit-risk-ann-ci.yml
```

The optional model smoke workflow installs deployment dependencies and verifies
that the saved ANN and preprocessing artifacts can be loaded:

```text
.github/workflows/credit-risk-model-smoke.yml
```

## Future Improvements

- Validate on a public real-world dataset with a documented license.
- Add probability calibration comparison using Platt scaling or isotonic
  regression.
- Optimize thresholds using expected loss and approval capacity.
- Add segment fairness analysis and governance documentation.
- Compare ANN performance with logistic regression, gradient boosting, and
  calibrated tree models.
- Add SHAP explanations with carefully designed, non-causal language.
- Track drift, calibration decay, and stability over time.
- Add model registry, API serving, and automated deployment tests.

## Skills Demonstrated

- Artificial Neural Networks for tabular data
- Binary classification and probability scoring
- Imbalanced-learning strategies
- Reproducible preprocessing pipelines
- Feature engineering
- Validation-based threshold tuning
- ROC and precision-recall analysis
- Probability calibration assessment
- Explainability using permutation importance
- Model persistence and reusable inference
- Streamlit application development
- Manual and batch scoring workflows
- Unit testing and GitHub Actions
- Deployment-ready ML engineering
- Responsible AI and model-governance framing

## Portfolio Positioning

**One-line description:** ANN-based credit risk engine that produces
probability-of-default scores, risk bands, and review recommendations through an
interactive Streamlit application.

**Pinned repository description:** End-to-end tabular deep-learning project with
reproducible preprocessing, class-weighted ANN training, probability calibration
metrics, threshold tuning, explainability, and deployable batch scoring.

This project supports a transition from Quality Data Scientist to broader Data
Science / ML / AI roles by showing that the same strengths used in quality
analytics—risk identification, structured root-cause thinking, metric
interpretation, automation, and stakeholder-oriented decisions—can be applied
to a governed predictive-modeling workflow.

## Responsible Use

This repository is a portfolio demonstration. It is not validated for
production underwriting and does not assess legal compliance, bias, protected
classes, or adverse-action obligations.

## Author

**Anmol Tripathi**  
Quality Data Scientist transitioning toward Data Science, Machine Learning,
Applied AI, Analytics Engineering, and Quality Analytics roles.
