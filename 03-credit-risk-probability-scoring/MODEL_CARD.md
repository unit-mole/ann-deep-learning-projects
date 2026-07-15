# Model Card — Credit Risk Probability Scoring ANN

## Model Overview

| Field | Value |
|---|---|
| Model name | Credit Risk Probability Scoring ANN |
| Model version | 1.0.0 |
| Model owner | Anmol Tripathi |
| Last local validation | 2026-07-14 |
| Framework | TensorFlow / Keras |
| Model type | Feed-forward artificial neural network |
| Task | Binary default-risk classification with probability scoring |
| Deployment | Streamlit Community Cloud |
| Intended status | Educational portfolio demonstration |

## Intended Use

The model demonstrates an end-to-end credit-risk analytics workflow:

- Produce a probability-of-default style score.
- Assign Low, Medium, or High Risk categories.
- Generate review recommendations and priorities.
- Support manual, sample, and CSV batch scoring.
- Demonstrate reproducible preprocessing and model persistence.

## Prohibited Use

The model must not be used for:

- Real lending approval or rejection.
- Pricing, credit-limit, or collections decisions.
- Adverse-action notices.
- Decisions involving protected classes.
- Production underwriting without independent validation, governance,
  compliance review, security controls, fairness analysis, and monitoring.

## Training Data

The project uses a deterministic synthetic banking dataset containing 25,000
applicant records.

Observed target distribution:

| Class | Records | Share |
|---|---:|---:|
| Non-default | 19,682 | 78.73% |
| Default | 5,318 | 21.27% |

The data contains no private customer records. Controlled missingness is added
to selected fields to demonstrate imputation.

## Input Feature Groups

### Financial capacity

- Annual income
- Monthly income
- Loan amount
- Interest rate
- Debt-to-income ratio
- Loan-to-income ratio

### Credit behavior

- Credit-history length
- Revolving utilization
- Delinquency count
- Inquiry count
- Open accounts
- Existing loans
- Credit grade

### Collateral and loan context

- Collateral value
- Collateral ratio
- Loan purpose

### Applicant and categorical context

- Age
- Employment length
- Home ownership
- Region
- Verification status

## Preprocessing

- Median imputation for numerical fields.
- Standard scaling for numerical fields.
- Most-frequent imputation for categorical fields.
- One-hot encoding for categorical fields.
- Training-derived preprocessing statistics stored in
  `models/preprocessing_schema.json`.
- Serialized preprocessing pipeline stored in `models/preprocessor.joblib`.

The final processed input dimension is 41.

## Model Architecture

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
Dense 1 + Sigmoid
```

Training uses Adam, binary cross-entropy, balanced class weights, early stopping,
and learning-rate reduction.

## Output Definition

The ANN returns a continuous score between 0 and 1.

- Higher values indicate higher modeled default risk.
- The score is not a validated real-world probability of default.
- Operational categories and recommendations are policy layers applied after
  model inference.

## Thresholds

### Binary classification threshold

```text
0.58
```

Selected on validation F1.

### Operational risk bands

| Score | Risk band |
|---|---|
| `< 0.20` | Low Risk |
| `0.20–<0.50` | Medium Risk |
| `>= 0.50` | High Risk |

These boundaries are demonstration policies and are not production lending
thresholds.

## Reference Test Performance

| Metric | Result |
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

## Explainability

Permutation importance is used for global model-performance interpretation.
Leading signals include loan-to-income ratio, debt-to-income ratio, monthly
income, delinquency count, revolving utilization, credit grade, home ownership,
and collateral value.

These results are:

- Global rather than applicant-specific.
- Associational rather than causal.
- Not sufficient for adverse-action explanations.
- Dependent on the synthetic data-generating process.

## Known Limitations and Failure Modes

- Synthetic data cannot represent real lender portfolios or macroeconomic
  conditions.
- Probability calibration may not transfer beyond the generated dataset.
- Category distributions may differ in real applications.
- Unknown categorical levels are encoded without learned category-specific
  effects.
- The ANN may be less interpretable than simpler baselines.
- Thresholds do not incorporate expected loss, approval capacity, fairness, or
  regulation.
- Model performance can degrade under data drift.
- The project has not undergone independent validation.

## Monitoring Recommendations

A production adaptation would require monitoring for:

- Input-data drift.
- Missing-value and category-frequency changes.
- Score-distribution drift.
- Calibration decay.
- Approval and review-rate changes.
- Precision, recall, and loss by portfolio segment.
- Fairness and disparate-impact indicators.
- Model-artifact and preprocessing-schema consistency.
- Delayed-outcome performance.

## Retraining Triggers

Potential retraining triggers include:

- Material drift in applicant or loan characteristics.
- Calibration deterioration.
- Changes in lending policy or product mix.
- Sustained performance decline.
- New data definitions or source-system changes.
- New regulatory or governance requirements.

## Artifact Locations

| Artifact | Location |
|---|---|
| Primary deployed ANN | `models/final_credit_risk_ann_model.keras` |
| Best validation checkpoint | `models/best_credit_risk_ann_model.keras` |
| Serialized preprocessor | `models/preprocessor.joblib` |
| Portable preprocessing schema | `models/preprocessing_schema.json` |
| Project metadata | `models/project_metadata.json` |
| Test metrics | `outputs/model_metrics.json` |
| Training history | `outputs/training_logs.csv` |
| Explainability output | `outputs/permutation_importance.png` |

## Validation Checklist

The portfolio release was validated by:

- Loading both Keras model artifacts.
- Confirming input shape `(None, 41)`.
- Confirming output shape `(None, 1)`.
- Running the end-to-end sample scoring pipeline.
- Running three automated tests successfully.
- Compiling application and source modules.
- Testing manual, sample, and batch Streamlit workflows.
- Deploying and testing the public Streamlit application.
- Running the GitHub Actions CI workflow successfully.

## Responsible-Use Statement

This model card documents an educational project. It does not establish
fitness for production lending, regulatory compliance, fairness, security, or
model-risk-management approval.
