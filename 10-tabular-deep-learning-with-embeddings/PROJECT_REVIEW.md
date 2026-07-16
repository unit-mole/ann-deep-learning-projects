# Project Review and Improvements

## Confirmed project type

The uploaded notebook, saved Keras model, preprocessing objects, and metric files
confirm that this is a **binary classification** project.

- Output activation: sigmoid
- Loss: binary cross-entropy
- Categorical inputs: 7
- Numerical inputs: 7
- Dataset size: 25,000 synthetic rows
- Saved model parameters: 15,649

## Strong parts retained

- Functional multi-input Keras architecture
- One embedding layer for each categorical variable
- Numerical imputation and scaling
- Class weighting and early stopping
- Random Forest baseline comparison
- ROC-AUC, PR-AUC, classification metrics, and threshold analysis
- Saved Keras model and preprocessing artifacts

## Corrections and production upgrades

1. **Removed categorical preprocessing leakage**
   - The original notebook fitted category encoders using combined train,
     validation, and test values.
   - The new training pipeline fits every encoder on training data only.

2. **Added safe unseen-category handling**
   - The retraining encoder reserves index 0 for out-of-vocabulary values.
   - The supplied legacy model uses documented fallback categories and reports
     unknown counts instead of failing.

3. **Separated notebook logic into reusable modules**
   - Data generation, preprocessing, model construction, evaluation, embedding
     analysis, and inference now live under `src/`.

4. **Added a complete inference layer**
   - Column validation, numeric coercion, fixed feature order, saved artifact loading,
     probabilities, classes, business buckets, and interpretations are centralized
     in `PredictionPipeline`.

5. **Added a hosted demo**
   - Manual single prediction
   - Included sample or uploaded CSV batch scoring
   - Prediction summary and charts
   - Unknown-category warning
   - Downloadable scored CSV
   - Model and embedding insights

6. **Strengthened reproducibility**
   - Deterministic synthetic data generator
   - Training-only preprocessing
   - Validation-only threshold selection
   - One-hot Random Forest baseline pipeline
   - Saved metadata and metrics
   - Pinned dependencies

7. **Added engineering safeguards**
   - pytest inference tests
   - GitHub Actions workflow
   - `.gitignore` for environments, generated data, secrets, logs, and temporary models
   - Streamlit and monorepo deployment instructions

## Portfolio interpretation of the result

At threshold 0.50, the Random Forest has slightly higher accuracy, while the
embedding ANN delivers substantially higher positive-class recall and F1. The honest
portfolio conclusion is therefore not that embeddings universally outperform tree
models. It is that the ANN provides a stronger positive-case discovery trade-off and
demonstrates a scalable architecture for mixed tabular data with categorical
variables.

## Validation performed on the packaged project

- Python source compilation completed successfully.
- Three inference/preprocessing tests passed.
- The saved Keras model loaded and produced valid probabilities.
- An unseen category was handled without an exception.
- The Streamlit application completed a headless startup smoke test.
