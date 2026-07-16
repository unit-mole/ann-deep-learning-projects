# Model Card

## Model

Shared-trunk, three-head artificial neural network for mixed-output customer analytics.

## Intended Use

Portfolio demonstration of multi-task learning, output-specific losses, deployment artifacts, and business-oriented scoring.

## Inputs

17 raw customer features. Two deterministic features—tenure cohort and spend band—are generated inside the pipeline. After imputation, scaling, and one-hot encoding, the ANN receives 42 values.

## Outputs

1. churn probability and thresholded binary label;
2. customer lifetime value;
3. engagement score from 0 to 100.

## Data

15,000 reproducibly generated synthetic customer records. No personal or proprietary data is included.

## Evaluation

See `outputs/model_metrics.json`. Regression performance is strong. Churn performance is constrained by approximately 2.17% positive prevalence; PR-AUC, precision, recall, F1, and ranking metrics should be considered together.

## Limitations

- synthetic relationships may not transfer to real customers;
- the bundled artifact was adapted from the supplied model by removing the leaked cluster input;
- predicted probabilities are not calibrated for production use;
- recommendation text is deterministic business logic, not causal advice;
- fairness, drift, and intervention effects have not been validated.

## Ethical Considerations

Do not use the model for credit, employment, insurance, healthcare, or other high-impact decisions. Any real deployment requires approved data, governance, fairness review, monitoring, and human oversight.
