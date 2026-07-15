# Delivery Notes

## Major changes made from the uploaded project

1. Reframed the task from predicting an engineered optimal-price target to predicting **demand at a proposed price**.
2. Removed the target-leaking segment dependency that used realized demand and optimal price as model inputs.
3. Rebuilt the saved models as portable Keras 3 artifacts compatible with the hosted inference environment.
4. Added decision-time data validation, missing-value handling, categorical normalization, range checks, and safe defaults.
5. Added calendar, competitor-gap, markup, inventory-pressure, and promotion-interaction features.
6. Added a tabular ANN regressor with categorical embeddings, batch normalization, dropout, Huber loss, and early stopping.
7. Added a supporting ANN high-demand classifier.
8. Added vectorized candidate-price scoring for revenue, margin, and balanced objectives.
9. Added transparent pricing guardrails and rule-based business recommendations.
10. Replaced target-derived clustering with transparent, inference-safe pricing segments.
11. Added held-out regression/classification metrics and portfolio evaluation plots.
12. Added within-category permutation importance to reduce unrealistic cross-category permutations.
13. Added a complete Streamlit app for manual and CSV batch optimization.
14. Added sample data, downloadable result support, tests, GitHub Actions CI, local-run instructions, and Streamlit deployment guidance.
15. Added documentation for data safety, original-model review, screenshots, project positioning, and limitations.

## Validated results

- Demand MAE: 12.25 units
- Demand RMSE: 15.68 units
- Demand R²: 0.667
- Demand MAPE: 14.41%
- High-demand classifier accuracy: 92.54%
- High-demand classifier F1: 0.890
- Test rows: 2,400
- Automated tests: 5 passed

## Files intentionally not included

The original uploaded legacy models and preprocessors are not used by the hosted application because their pipeline depends on leaked/future information and the saved models contain a legacy BatchNormalization configuration that is incompatible with the current Keras 3 runtime.

Temporary `training_context.joblib` and `evaluation_payload.joblib` files are also excluded. They are recreated automatically when the training command is run and are not needed for inference.
