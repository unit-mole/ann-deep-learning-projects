# Model Artifacts

- `multi_output_model.keras`: deployment artifact aligned with the 42-column saved preprocessor.
- `original_supplied_model.keras`: untouched original model retained for provenance.
- `preprocessor.joblib`: fitted imputation, scaling, and one-hot encoding pipeline.
- `target_metadata.json`: inverse-scaling values, feature-bin metadata, and churn threshold.
- `feature_schema.json`: required columns, categories, and defaults.

The deployment artifact removes the original target-derived cluster input. Run `python -m src.model_training` to replace it with a fully retrained leakage-safe model.
