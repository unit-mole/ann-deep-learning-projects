# Project Audit and Changes Made

## Uploaded source reviewed

- Notebook: `House Price Prediction using Artificial Neural Networks(1).ipynb`
- Dataset: `house_prices.csv`
- Saved ANN: `house_price_ann.keras`
- Scaler: `house_price_scaler.pkl`
- Best parameters: `house_price_best_params.json`

## Confirmed original implementation

- Dataset: California Housing
- Rows: 20,640
- Model inputs: 8 numeric features
- Split: 70% train, 15% validation, 15% test
- Preprocessing: StandardScaler fitted on training data
- Best architecture: 128-unit ReLU layer, 20% dropout, 64-unit ReLU layer, linear output
- Uploaded test performance:
  - MAE: 0.3503 target units ≈ $35,035
  - RMSE: 0.5116 target units ≈ $51,164
  - R²: 0.8020
  - MAPE: 19.39%

## Technical issues corrected

1. Removed classification-only imports and framing from the regression workflow.
2. Replaced machine-specific absolute paths with project-relative paths.
3. Converted the notebook into modular reusable Python files.
4. Added schema validation and exact model feature ordering.
5. Added future-safe median imputation based on training statistics.
6. Added IQR outlier reporting without blindly deleting valid geographic extremes.
7. Added MAE, RMSE, R², MAPE, actual-vs-predicted, residual, and error analyses.
8. Added global permutation importance.
9. Added local sensitivity drivers for individual predictions.
10. Added empirical 80% error ranges and distribution-relative price categories.
11. Added single and batch Streamlit prediction workflows.
12. Added downloadable CSV output.
13. Added model metadata for feature ranges, units, metrics, thresholds, and limitations.
14. Added tests, CI workflow, hosting documentation, and screenshot guidance.
15. Corrected the business scope: the dataset predicts block-group median value,
    not the transaction price of an individual house.
16. Documented the source target ceiling near $500,001.
17. Preserved the actual supplied ANN, scaler, and tuned parameters.
