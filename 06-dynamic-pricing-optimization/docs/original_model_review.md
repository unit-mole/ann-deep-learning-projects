# Review of the Original Dynamic Pricing Notebook

## What was retained

The original notebook demonstrated several strong portfolio ideas:

- synthetic retail pricing data with product, market, competition, inventory, promotion, and seasonality variables;
- numerical scaling and categorical encoding;
- a multi-input ANN with categorical embeddings;
- separate regression and classification outputs;
- saved Keras and Joblib artifacts;
- scenario-oriented pricing outputs.

## Critical issue corrected

The original regression pipeline created a customer/pricing segment from `realized_demand` and `optimal_price`, then supplied that segment to the ANN predicting `optimal_price`. Both values are future/target-derived information that would not exist when a live price must be selected. This is target leakage and produces optimistic validation results while making the pipeline impossible to use honestly in production.

The original saved Keras models also used an older BatchNormalization configuration containing a `renorm` argument that is not accepted by the current Keras 3 runtime used for this rebuild.

## New deployable formulation

1. **Target:** observed demand at the historical/proposed price.
2. **Inputs:** only decision-time variables available before the price is published.
3. **ANN prediction:** expected demand for each proposed candidate price.
4. **Optimization:** calculate expected revenue and gross margin for every candidate.
5. **Selection:** choose the highest-scoring price for the selected objective.
6. **Guardrails:** apply cost floor, controlled price-change limits, low-inventory protection, and competitor-gap checks.
7. **Segmentation:** assign transparent business segments after prediction; no target-derived cluster is fed into the ANN.

## Result

The new metrics are lower than the leaked notebook metrics, but they are materially more credible. The holdout demand model reports MAE 12.25 units, RMSE 15.68 units, R² 0.667, and MAPE 14.41%. These are the metrics displayed in the portfolio README and app.
