# Model Artifacts

| File | Purpose |
|---|---|
| `credit_card_fraud_ann.keras` | Saved Keras ANN used by the demo |
| `credit_card_scaler.pkl` | StandardScaler fitted on the training split |
| `credit_card_best_params.json` | Supplied best hyperparameters |
| `feature_schema.json` | Ordered production feature contract |
| `decision_threshold.json` | Default and validation-derived thresholds |

## Important

The included `.keras` model and scaler are the supplied artifacts audited from
the original project. They make the Streamlit demo runnable without retraining.

The improved training code in `src/model_training.py`:

- computes balanced class weights from the training split;
- monitors validation PR-AUC;
- rebuilds a fresh final model;
- selects the operating threshold using validation data;
- overwrites these artifacts when retraining succeeds.

Do not edit `feature_schema.json` without retraining the scaler and model.
Feature names and ordering must remain identical across training and inference.

## Security

Python pickle files should only be loaded from a trusted source. Never replace
`credit_card_scaler.pkl` with a pickle downloaded from an unknown location.
