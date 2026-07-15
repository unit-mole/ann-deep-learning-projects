# Changes from the original notebook

- Replaced direct scaling of physiologically implausible zeros with zero-to-missing conversion and train-only median imputation.
- Saved one preprocessing pipeline containing both imputation and scaling instead of saving a raw `StandardScaler` alone.
- Added missing-value indicators for the five affected medical fields.
- Added balanced class weights to address the 500/268 class distribution.
- Added validation-only threshold selection using F2 to prioritize recall for a screening demonstration.
- Separated transparent Low/Medium/High communication bands from the model's classification threshold.
- Added PR-AUC, precision–recall curve, risk distribution, threshold analysis, and permutation importance.
- Added a reusable prediction pipeline for both single-record and batch inference.
- Added a Streamlit app, healthcare disclaimers, tests, CI, hosting instructions, and recruiter-focused documentation.
- Retrained the ANN because the original saved model was paired with a scaler fitted on raw zero markers and therefore could not be safely combined with the improved preprocessing logic.

## Portfolio hardening and deployment updates

- Added strict validation for missing columns, empty batches, nonnumeric values, infinite values, and negative measurements.
- Added friendly Streamlit messages for unreadable, empty, or malformed CSV uploads.
- Added non-blocking warnings when manual or uploaded values fall outside the ranges observed in the model-development dataset; scoring continues, but the app flags the extrapolation risk.
- Added public Streamlit Community Cloud deployment details and verified anonymous access.
- Added dataset provenance, attribution, responsible-use, and licensing notes.
- Explicitly pinned Keras `3.15.0` alongside TensorFlow `2.20.0` for reproducible local and cloud environments.
- Added a medical browser-tab icon (`🩺`) and retained the healthcare disclaimer throughout the application.
- Renamed the batch-output screenshot to `05_batch_scored_output.png` and updated its README reference.
- Expanded preprocessing tests to cover invalid text, negative values, empty inputs, and out-of-training-range warnings.

## Baseline comparison on the same test split

| Version | Accuracy | Positive precision | Positive recall | Positive F1 | ROC-AUC | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| Original notebook | 0.750 | 0.731 | 0.463 | 0.567 | 0.854 | 22 |
| Portfolio pipeline | 0.741 | 0.593 | 0.854 | 0.700 | 0.856 | 6 |

The portfolio pipeline uses a validation-selected threshold of 0.33 and is intentionally optimized toward recall.
