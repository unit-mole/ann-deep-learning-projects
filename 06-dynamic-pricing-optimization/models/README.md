# Model Artifacts

The hosted application requires these committed artifacts:

- `dynamic_pricing_demand_ann.keras` — Keras 3 ANN that predicts log-demand and is converted back to demand units during inference.
- `dynamic_pricing_demand_state_ann.keras` — supporting ANN classifier that estimates high-demand probability.
- `numeric_scaler.joblib` — numerical feature scaler fitted on training data only.
- `model_metadata.json` — feature schema, categorical encoder maps, threshold, and model version.
- `model_metrics.json` — validated holdout metrics shown in the README and app.

The `.keras` files are portable Keras model archives. Training uses the JAX backend; the Streamlit app explicitly selects the PyTorch backend before importing Keras.

`training_context.joblib` and `evaluation_payload.joblib` are temporary files generated during retraining and are excluded from Git. They are not required for hosted inference.
