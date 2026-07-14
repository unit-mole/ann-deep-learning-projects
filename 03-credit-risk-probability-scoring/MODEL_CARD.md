# Model Card — Credit Risk Probability Scoring ANN

## Intended use

Educational portfolio demonstration of an ANN-based probability scoring workflow for tabular credit-risk data.

## Out-of-scope use

Do not use this model for real credit approval, adverse-action notices, pricing, or other consequential decisions. The training data is synthetic and does not establish fairness, legal compliance, stability, or performance on any real population.

## Training data

25,000 deterministic synthetic applicant records with a 21.27% default class. Six columns receive 1% missingness to exercise the preprocessing pipeline.

## Model

Dense ANN: 256 → 128 → 64 → 32 hidden units with ReLU, L2 regularization, batch normalization, dropout, and a sigmoid output.

## Evaluation

Test ROC-AUC 0.813, PR-AUC 0.610, recall 0.649, precision 0.477, F1 0.550, and Brier score 0.174 at the validation-selected classification threshold of 0.58.

## Limitations

Synthetic-data realism, probability calibration, threshold economics, fairness across protected groups, reject inference, population drift, and regulatory governance have not been validated.
