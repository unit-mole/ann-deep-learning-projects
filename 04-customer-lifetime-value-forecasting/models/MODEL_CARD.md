# Model Card

## Model

**Name:** CLV ANN with categorical embeddings  
**Framework:** Keras 3.14.0  
**Tasks:** 90-day future-revenue regression and 90-day retention classification  
**Parameters:** 77,346  
**Data:** Synthetic customer transactions

## Intended Use

- Portfolio demonstration of tabular deep learning
- Customer-value segmentation prototypes
- Marketing and retention decision-support demonstrations
- Batch and single-customer scoring through Streamlit

## Out-of-Scope Use

- Credit, insurance, employment, housing, healthcare, or other high-impact decisions
- Production financial forecasts without real-data retraining and validation
- Individual targeting using sensitive personal attributes

## Inputs

Nineteen numerical features and five categorical features. See `models/model_metadata.json` for the exact schema and category values.

## Outputs

1. Predicted future revenue over the next 90 days
2. Predicted probability of at least one future order during the same period

The app converts the revenue output into a value segment and combines it with retention probability to generate a business action.

## Evaluation

| Metric | Value |
|---|---:|
| MAE | $103.68 |
| RMSE | $144.01 |
| R² | 0.0924 |
| Retention accuracy | 0.7642 |
| Retention F1 | 0.8434 |
| Retention ROC-AUC | 0.7308 |

## Limitations

- The data is synthetic.
- Revenue fit is modest, and the highest-value customers are underpredicted.
- The original cluster estimator was not saved.
- Segment thresholds are distribution-dependent.
- Model uncertainty is not estimated.

## Recommended Production Controls

- Out-of-time validation and champion/challenger benchmarking
- Fairness and privacy review
- Drift, calibration, and segment-stability monitoring
- Human review for high-value actions
- Experiment-based measurement of incremental business lift
