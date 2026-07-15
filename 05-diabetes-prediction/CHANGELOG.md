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

## Baseline comparison on the same test split

| Version | Accuracy | Positive precision | Positive recall | Positive F1 | ROC-AUC | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| Original notebook | 0.750 | 0.731 | 0.463 | 0.567 | 0.854 | 22 |
| Portfolio pipeline | 0.741 | 0.593 | 0.854 | 0.700 | 0.856 | 6 |

The portfolio pipeline uses a validation-selected threshold of 0.33 and is intentionally optimized toward recall.
