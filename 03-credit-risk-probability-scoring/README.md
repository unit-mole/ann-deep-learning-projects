# Credit Risk Probability Scoring using Artificial Neural Networks

[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ANN-FF6F00)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end credit analytics portfolio project that converts applicant data into a probability of default, operational risk category, and review recommendation using a deep neural network.

**Live demo:** `https://<your-app-name>.streamlit.app`  
**Notebook:** [`notebooks/credit_risk_probability_scoring.ipynb`](notebooks/credit_risk_probability_scoring.ipynb)

## Business problem

Lenders need more than a binary approve/reject prediction. They need a comparable risk score that can rank applicants, support manual-review queues, and make decision policies transparent. This project answers:

> Given applicant information, what is the probability that the applicant belongs to a higher default-risk category?

The deployed pipeline returns:

- **Risk probability score**
- **Low / Medium / High risk category**
- **Decision recommendation and review priority**

## Portfolio scope

This is an educational demonstration built on a deterministic **synthetic banking dataset**. It is not a production credit model and must not be used for real lending decisions.

## Dataset

The notebook generates 25,000 applicant records with demographic, financial, credit-history, collateral, and loan-purpose fields. The observed target distribution is:

| Class | Records | Share |
|---|---:|---:|
| Non-default | 19,682 | 78.73% |
| Default | 5,318 | 21.27% |

Six fields receive 1% controlled missingness to demonstrate imputation. No private customer data is included in GitHub.

## Workflow

1. Generate or load applicant-level data.
2. Create `monthly_income`, `loan_to_income`, and `collateral_ratio`.
3. Split data into 70% train, 15% validation, and 15% test sets with stratification.
4. Median-impute and standardize numerical fields.
5. Most-frequent-impute and one-hot encode categorical fields.
6. Handle imbalance with balanced class weights.
7. Train an ANN with early stopping, learning-rate reduction, and best-model checkpointing.
8. Select the binary classification threshold on validation F1.
9. Evaluate probability quality, classification performance, calibration, and error trade-offs.
10. Serve single and batch scoring through Streamlit.

## Feature engineering

| Derived field | Formula | Purpose |
|---|---|---|
| Monthly income | Annual income / 12 | Monthly affordability proxy |
| Loan-to-income | Loan amount / annual income | Relative debt burden |
| Collateral ratio | Collateral value / loan amount | Security coverage proxy |

The portable preprocessing schema stores training medians, scaling statistics, categorical modes, category order, and final feature order. This makes the demo less dependent on scikit-learn pickle compatibility.

## ANN architecture

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

Training uses Adam, binary cross-entropy, ROC-AUC, PR-AUC, balanced class weights, early stopping, and `ReduceLROnPlateau`.

## Probability and decision logic

The ANN outputs a continuous probability between 0 and 1. Two threshold layers are intentionally separated:

- **Classification threshold: 0.58** — selected on the validation set to maximize F1.
- **Operational risk bands:**
  - `< 0.20`: Low Risk
  - `0.20–<0.50`: Medium Risk
  - `>= 0.50`: High Risk

Risk bands are a policy layer for demonstration. A real lender would set them using expected-loss economics, approval capacity, portfolio appetite, calibration, fairness review, and regulation—not validation F1 alone.

## Model results

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
| Actual non-default | 2,384 | 568 |
| Actual default | 280 | 518 |

A **false negative** is a risky applicant treated as lower risk and can create credit loss. A **false positive** is a safer applicant flagged as risky and can reduce approvals, revenue, and customer access. Threshold selection therefore requires business-cost analysis, not accuracy alone.

## Class imbalance

Defaults are only 21.27% of the synthetic data. The notebook uses balanced class weights (`0: 0.635`, `1: 2.351`) rather than generating synthetic minority examples. This retains original training rows while making default errors more influential. Precision-recall analysis and validation threshold tuning complement the class weighting.

## Explainability

Permutation importance identifies the strongest model-performance drivers in the executed notebook. The leading signals include loan-to-income ratio, debt-to-income ratio, monthly income, delinquency count, revolving utilization, grade, home ownership, and collateral value. These are global associations—not causal explanations or legally sufficient adverse-action reasons.

![Permutation importance](outputs/permutation_importance.png)

## Visual results

| Confusion matrix | ROC curve |
|---|---|
| ![Confusion matrix](outputs/confusion_matrix.png) | ![ROC curve](outputs/roc_curve.png) |

| Precision-recall curve | Calibration curve |
|---|---|
| ![PR curve](outputs/precision_recall_curve.png) | ![Calibration curve](outputs/calibration_curve.png) |

| Training AUC | Risk bands |
|---|---|
| ![Training AUC](outputs/training_auc.png) | ![Risk category distribution](outputs/risk_category_distribution.png) |

## Streamlit demo

The app supports:

- Manual applicant entry
- CSV upload for batch scoring
- Preloaded safe sample data
- Probability, category, predicted class, and recommendation
- Risk distribution chart
- Downloadable scored CSV

A screenshot should be saved as `images/demo_screenshot.png` after deployment.

## Run locally

```bash
cd ANN/Credit_Risk_Probability_Scoring
python -m venv .venv
```

Windows:

```bash
.venv\Scriptsctivate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Retrain the model

The included model is ready for the demo. To reproduce training on the synthetic dataset:

```bash
python -m src.model_training
```

To train on a compatible CSV containing `default_flag`:

```bash
python -m src.model_training --data path/to/training_data.csv
```

The script writes model checkpoints to `models/` and training logs to `outputs/`.

## Deploy

Streamlit Community Cloud is the recommended host because the application is already Streamlit-based and can deploy directly from GitHub. Set the entrypoint to:

```text
Credit_Risk_Probability_Scoring/app/streamlit_app.py
```

See [`README_HOSTING.md`](README_HOSTING.md) for detailed Streamlit and Hugging Face Docker instructions.

## Project structure

```text
Credit_Risk_Probability_Scoring/
├── app/streamlit_app.py
├── data/
│   ├── README_data.md
│   └── sample_input.csv
├── notebooks/credit_risk_probability_scoring.ipynb
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── risk_scoring.py
│   └── prediction_pipeline.py
├── models/
│   ├── final_credit_risk_ann_model.keras
│   ├── best_credit_risk_ann_model.keras
│   ├── preprocessor.joblib
│   ├── preprocessing_schema.json
│   └── project_metadata.json
├── outputs/
├── images/
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
├── MODEL_CARD.md
├── README_HOSTING.md
├── requirements.txt
└── README.md
```

## Future improvements

- Validate on a public real-world dataset with a documented license.
- Add probability calibration comparison using Platt scaling or isotonic regression.
- Optimize thresholds using expected loss and approval capacity.
- Add segment fairness analysis and governance documentation.
- Compare ANN performance with logistic regression, gradient boosting, and calibrated tree models.
- Add SHAP explanations with carefully designed, non-causal language.
- Track drift, calibration decay, and stability over time.
- Add model registry, API serving, and automated deployment tests.

## Skills demonstrated

Artificial neural networks, binary classification, probability scoring, imbalanced learning, preprocessing pipelines, feature engineering, threshold tuning, ROC/PR analysis, calibration, explainability, modular Python, Streamlit, testing, CI, Docker, model documentation, and responsible AI framing.

## Portfolio positioning

**One-line description:** ANN-based credit risk engine that produces probability-of-default scores, risk bands, and review recommendations through an interactive Streamlit app.

**Pinned repository description:** End-to-end tabular deep-learning project with reproducible preprocessing, class-weighted ANN training, probability calibration metrics, threshold tuning, explainability, and deployable batch scoring.

This project supports a transition from Quality Data Scientist to broader Data Science / ML / AI roles by showing that the same strengths used in quality analytics—risk identification, structured root-cause thinking, metric interpretation, automation, and stakeholder-oriented decisions—can be applied to a governed predictive modeling workflow.

## Responsible-use notice

This repository is a portfolio demonstration. It is not validated for production underwriting and does not assess legal compliance, bias, protected classes, or adverse-action obligations.
