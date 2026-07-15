# Technical Review of the Supplied CLV Project

## Review Scope

This project package was created from the supplied notebook and trained artifacts:

- `Customer_Lifetime_Value_Forecasting_ANN_COMPLETE(1).ipynb`
- `clv_ann_model.keras`
- `numeric_scaler.pkl`
- `label_encoders.pkl`
- `test_predictions.csv`
- `training_history.csv`

The original notebook is preserved unchanged in `notebooks/archive/` for traceability. A cleaner modular notebook is provided at `notebooks/customer_lifetime_value_forecasting.ipynb`.

## What the Original Project Already Did Well

- Generated a privacy-safe synthetic transaction dataset.
- Created a customer snapshot using separate observation and 90-day forecast windows.
- Engineered meaningful RFM, cohort, engagement, diversity, discount, and rate features.
- Used a genuine artificial neural network for tabular data rather than relabeling a conventional model as ANN.
- Used learned embeddings for five categorical inputs.
- Trained a multi-task network with a future-revenue regression head and a retention-classification head.
- Saved the trained Keras model, numerical scaler, categorical encoders, test predictions, and training history.

## Findings from the Actual Artifacts

### 1. The target is a 90-day revenue forecast

The supervised regression target is revenue generated during the next 90 days. The portfolio wording has therefore been made precise: this is a **90-day CLV proxy**, not an unlimited-horizon lifetime value estimate.

### 2. The saved model is a multi-task ANN

The model contains 19 numerical inputs, five categorical embedding inputs, shared dense layers, a non-negative revenue output, and a retention-probability output. It contains 77,346 parameters.

### 3. The original categorical encoding had minor leakage

The supplied label encoders were fitted after concatenating categories from the training, validation, and test splits. This does not expose targets, but it does expose held-out category vocabulary. The new training pipeline fits encoders on training data only and maps unseen categories safely.

### 4. Customer-cluster artifacts were not saved

The original notebook used K-means clusters as `customer_segment_name`, but the cluster scaler and K-means estimator were not included with the deployment files. The supplied model still requires that categorical input.

The deployment pipeline handles this in two ways:

1. It uses `customer_segment_name` directly when the field is supplied.
2. It applies a documented deterministic fallback heuristic when the field is absent.

The clean retraining pipeline saves both `segment_scaler.pkl` and `segment_kmeans.pkl`, removing this limitation for future model versions.

### 5. Standard MAPE is not a reliable headline metric

About 24.9% of test customers have zero actual future revenue. Standard MAPE divides by the actual value and becomes undefined or extremely unstable near zero. The project therefore reports:

- MAE
- RMSE
- R²
- MAPE on non-zero actuals only
- sMAPE
- WAPE

### 6. The supplied model is a valid baseline, not a production model

Recalculated test performance from the supplied predictions:

| Metric | Result |
|---|---:|
| MAE | $103.68 |
| RMSE | $144.01 |
| R² | 0.0924 |
| sMAPE | 87.65% |
| WAPE | 71.79% |
| Retention accuracy | 0.7642 |
| Retention F1 | 0.8434 |
| Retention ROC-AUC | 0.7308 |

The regression model explains modest variance and underpredicts some high-value customers. The README and app present this honestly instead of overstating model quality.

## Improvements Included in the Portfolio Version

- Professional numbered monorepo folder name: `04-customer-lifetime-value-forecasting`.
- Modular preprocessing, feature engineering, training, evaluation, scoring, and inference files.
- A Streamlit app for manual and batch scoring.
- Automatic derived-feature calculation and input validation.
- Unknown-category handling for safer inference.
- Business value segments and action recommendations.
- Downloadable scored CSV output.
- Regression, residual, distribution, training, and retention visualizations.
- A model card and machine-readable metadata file.
- Privacy-safe sample input data and data-governance notes.
- Unit tests and a GitHub Actions workflow.
- A corrected retraining pipeline that:
  - performs customer-level train/validation/test splitting,
  - fits preprocessing only on training data,
  - saves the segmentation estimator and scaler,
  - trains revenue on a log-transformed target using Huber loss,
  - selects thresholds from validation predictions,
  - creates permutation-importance output,
  - regenerates evaluation artifacts.
- Exact dependency pins for the saved model and preprocessing artifacts where compatibility matters.

## Validation Performed

- All Python source files compile successfully.
- The included unit-test suite passes.
- The sample CSV successfully produces all 19 numerical inputs and five categorical inputs expected by the model.
- The supplied Keras model was deserialized and used for inference with Keras 3.14.0.

The execution environment used for packaging did not include the TensorFlow backend, so the deserialization/inference smoke test used Keras 3.14.0 with an available backend. The deployment requirements pin TensorFlow 2.20.0 and Keras 3.14.0 for Streamlit Community Cloud.

## Recommended Portfolio Interpretation

Present this project as an end-to-end customer-analytics and model-deployment project. Its strongest evidence is the full workflow—customer-level feature engineering, multi-input ANN design, dual-task prediction, honest evaluation, business segmentation, modular inference, Streamlit deployment, testing, and CI—not a claim that the current synthetic-data baseline is ready for production use.
