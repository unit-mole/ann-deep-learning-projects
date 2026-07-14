# Audit of the Original Project

## Files Reviewed

The GitHub-ready version was built from the supplied project assets:

- Original Jupyter notebook
- Full `creditcard.csv`
- Saved Keras ANN model
- Saved StandardScaler
- Best-parameter JSON

The original notebook is preserved unchanged at:

```text
notebooks/archive/original_credit_card_fraud_detection.ipynb
```

A cleaned portfolio notebook is provided separately.

## What Was Already Strong

The original project correctly included:

- the expected 284,807-row public dataset;
- the full 30-feature schema;
- a stratified train/validation/test split;
- a StandardScaler fitted on the training split;
- an ANN with two hidden ReLU layers and dropout;
- sigmoid output and binary cross-entropy loss;
- Adam optimization;
- precision and recall tracking;
- ROC-AUC evaluation;
- a confusion matrix;
- model and scaler persistence;
- fraud-probability predictions.

The supplied model architecture contains:

- Dense layer with 64 units
- Dropout at 20%
- Dense layer with 32 units
- Dropout at 20%
- Sigmoid output
- 4,097 trainable parameters

## Issues Identified and Corrected

### 1. Imbalance handling was described but not applied

The notebook text referred to class weighting or resampling, but the training
calls did not pass `class_weight` and did not apply a resampling method.

**Correction:** the improved training module computes balanced class weights
from the training labels and passes them to `model.fit`.

### 2. Accuracy was still prominent for an extreme-imbalance problem

The original evaluation included precision, recall, F1, ROC-AUC, and a
confusion matrix, but it did not calculate PR-AUC.

**Correction:** the project now reports PR-AUC / average precision and saves a
precision-recall curve.

### 3. The threshold was fixed at 0.50

Fraud predictions were created only with:

```python
test_probs >= 0.50
```

This did not demonstrate the operational precision-recall trade-off.

**Correction:** threshold-selection utilities now support:

- maximum validation F1;
- a validation recall target;
- an adjustable threshold in Streamlit.

The test set is not used to choose the threshold.

### 4. The selected tuning model was trained again without rebuilding

The hyperparameter-search candidate stored as `best_model` had already been
trained. The notebook then called `fit` again on the same model for final
training. This continues from previously learned weights instead of training a
fresh final model from the selected configuration.

**Correction:** the improved pipeline rebuilds a new ANN from the selected
hyperparameters before final training.

### 5. Silent synthetic-data fallback reduced reproducibility

The original data-loading cell downloaded the public dataset and silently
created a synthetic dataset if loading failed. This could cause two users to
train different projects under the same title without a clear failure.

**Correction:** the production training pipeline requires an explicit local
dataset and gives a clear setup error. No synthetic substitution occurs.

### 6. Several unused imports were present

The original notebook imported regression metrics, encoders, imputers, and
pipeline classes that were not used by this classification workflow.

**Correction:** the modular code imports only the required dependencies.

### 7. No deployable inference interface existed

The notebook could reload artifacts and score five rows, but it did not provide
a reusable schema-validated prediction service or user interface.

**Correction:** the project now includes:

- `src/prediction_pipeline.py`
- `app/streamlit_app.py`
- batch CSV prediction
- manual one-row scoring
- sample data
- summary metrics
- prediction export

### 8. Artifact metadata was incomplete

Only the model, scaler, and best parameters were saved.

**Correction:** the project adds:

- ordered feature schema;
- decision-threshold configuration;
- complete model metrics;
- classification report;
- threshold analysis;
- evaluation plots.

### 9. Full training data was not separated from deployment assets

The full CSV is approximately 151 MB and is unnecessary for the hosted app.

**Correction:** the GitHub package excludes the full CSV, includes a curated
sample, and documents how to obtain the source dataset.

### 10. Repository documentation was missing

The original project was notebook-centric.

**Correction:** the package now includes:

- recruiter-facing README;
- local setup instructions;
- deployment guide;
- data documentation;
- model documentation;
- output documentation;
- portfolio-positioning guidance;
- tests and CI configuration.

## Reference Artifact Audit

At the original 0.50 threshold, the supplied model produced:

| Metric | Value |
|---|---:|
| Accuracy | 99.9415% |
| Precision | 87.6923% |
| Recall | 77.0270% |
| F1-score | 82.0144% |
| ROC-AUC | 97.4227% |
| PR-AUC | 77.7964% |

Confusion matrix:

```text
TN = 42,640
FP = 8
FN = 17
TP = 57
```

These numbers were reproduced using the original notebook's split and the
supplied model/scaler artifacts.
