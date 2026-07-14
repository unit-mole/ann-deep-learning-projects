# Model Card: ANN Churn Classification

## Model Purpose

Estimate customer churn probability for educational and portfolio
demonstration purposes.

## Model Type

Feed-forward artificial neural network for binary classification.

## Legacy Architecture

- 12 processed input features
- Dense layer with 64 ReLU units
- Dense layer with 16 ReLU units
- One sigmoid output

## Legacy Evaluation

| Metric | Value |
|---|---:|
| Accuracy | 0.8565 |
| Churn precision | 0.7008 |
| Churn recall | 0.4707 |
| Churn F1 | 0.5632 |
| ROC-AUC | 0.8562 |
| PR-AUC | 0.6763 |

Confusion matrix:

```text
                 Predicted retained   Predicted churn
Actual retained          1528               79
Actual churn              208              185
```

## Appropriate Use

- Demonstrating ANN classification
- Comparing decision thresholds
- Learning deployment and batch prediction
- Exploring retention prioritization workflows

## Inappropriate Use

- Fully automated customer treatment decisions
- Credit, lending, eligibility, or pricing decisions
- Production use without calibration, fairness testing, monitoring, and governance

## Limitations

- The dataset may not represent a current production population.
- The churn class is imbalanced.
- The legacy test set was used for early-stopping validation.
- The model does not establish why a customer may churn.
- Probability calibration has not yet been formally assessed.
- Fairness across demographic and geographic groups requires analysis.
- Model performance may degrade when customer behavior changes.

## Recommended Next Improvements

1. Establish a logistic-regression and tree-based baseline.
2. Use a separate stratified train/validation/test split.
3. Tune the threshold against retention costs.
4. Add calibration curves and Brier score.
5. Add subgroup performance checks.
6. Add feature attribution or sensitivity analysis.
7. Add data-drift and prediction-drift monitoring.
