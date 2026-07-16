# Tabular Deep Learning with Embeddings

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-3.14-D00000.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B.svg)](https://streamlit.io/)
[![Task](https://img.shields.io/badge/Task-Binary%20Classification-6A5ACD.svg)](#problem-definition)

> A multi-input neural network that learns dense representations for categorical
> business variables, combines them with scaled numerical features, and produces an
> interpretable positive-outcome propensity score.

**Live demo:** `https://YOUR-APP-NAME.streamlit.app`  
**Portfolio repository:** `https://github.com/unit-mole/ann-deep-learning-projects`

---

## Business problem

Structured business datasets commonly combine numerical behavior with categorical
attributes such as region, job role, product category, device type, and membership
tier. One-hot encoding treats each category as unrelated and can create very wide
feature matrices as cardinality grows.

This project answers:

> **How can deep learning handle mixed tabular business data more effectively by
> learning compact embeddings for categorical variables?**

The application demonstrates a deployable architecture for customer propensity,
risk ranking, conversion prediction, prioritization, or other binary business
classification workflows.

## Problem definition

The supplied notebook and artifacts implement **binary classification**.

- Output layer: one neuron with `sigmoid` activation
- Loss: binary cross-entropy
- Primary ranking metric: ROC-AUC
- Minority-class decision metrics: recall and F1
- Default decision threshold: 0.50
- Dataset: 25,000 deterministic synthetic records
- Positive-class share: 36.73%

The target represents a **synthetic positive business-outcome propensity**. It is not
real customer data and should not be used for real eligibility, credit, employment,
or safety decisions.

## Tools and technologies

- Python 3.12, pandas, NumPy, and scikit-learn
- Keras Functional API with TensorFlow runtime
- categorical embedding, concatenation, batch normalization, and dropout layers
- stratified train/validation/test splitting and class weighting
- ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, and log loss
- PCA embedding projection and raw-feature permutation importance
- Streamlit and Plotly for interactive single and batch inference
- pytest and GitHub Actions for automated validation

## Key results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Embedding ANN | 0.6706 | 0.5436 | **0.6445** | **0.5898** | **0.7294** | 0.5994 |
| Random Forest | **0.6772** | **0.6756** | 0.2335 | 0.3471 | 0.7225 | **0.5961** |

The Random Forest is marginally better on raw accuracy, but the embedding ANN is much
stronger at finding the positive class: recall improves from 23.35% to 64.45% and F1
improves from 34.71% to 58.98%. This makes the ANN more useful when missing positive
cases is costly and probabilities are used for ranking or prioritization.

Additional ANN results at threshold 0.50:

- PR-AUC: 0.5983
- Confusion matrix: `[[2169, 994], [653, 1184]]`
- Parameters: 15,649
- Validation-selected best-F1 threshold: approximately 0.34

Thresholds must be selected from validation data and aligned with business costs;
they should not be optimized on the final test set.

## Why embeddings are useful

For each categorical feature, the network learns a small trainable vector. Categories
that contribute similarly to the prediction can move closer together in embedding
space. Compared with one-hot encoding, embeddings can:

- reduce dimensionality for higher-cardinality features;
- learn similarity between categories;
- support nonlinear interactions with numerical features;
- provide reusable vectors for visualization and nearest-category analysis.

The project uses the documented rule:

```text
embedding_dimension = min(50, ceil(sqrt(vocabulary_size)) + 1)
```

This grows sub-linearly with vocabulary size and avoids arbitrary oversized vectors.
The current feature vocabularies are small, so learned dimensions range from 3 to 5.

## Data and features

### Categorical inputs

`region`, `education_level`, `occupation`, `channel`, `device_type`,
`membership_tier`, `product_category`

### Numerical inputs

`age`, `years_experience`, `monthly_sessions`, `avg_basket_value`,
`days_since_last_activity`, `credit_utilization`, `income_estimate`

### Data safety

The full dataset is generated from code and is not committed. Only a small sample
input file is tracked. This keeps the repository safe for public hosting and shows how
to separate reproducible demo data from private production data.

## Architecture

```text
region ------------> Embedding -> Flatten --┐
education_level ----> Embedding -> Flatten --|
occupation ---------> Embedding -> Flatten --|
channel ------------> Embedding -> Flatten --|
device_type --------> Embedding -> Flatten --|--> Concatenate
membership_tier ----> Embedding -> Flatten --|        +
product_category ---> Embedding -> Flatten --┘   scaled numerical inputs
                                                     |
                                              Dense 128 + BN + Dropout
                                                     |
                                              Dense 64 + BN + Dropout
                                                     |
                                              Dense 32 + Dropout
                                                     |
                                               Sigmoid probability
```

![Model architecture](outputs/model_architecture.png)

## Workflow

1. Generate reproducible mixed-type synthetic data.
2. Create stratified train, validation, and test partitions.
3. Fit median imputation and standardization on training numerical data only.
4. Fit categorical encoders on training data only.
5. Reserve an out-of-vocabulary index in the retraining pipeline.
6. Create one embedding input per categorical feature.
7. Concatenate flattened embeddings with scaled numerical features.
8. Train with class weights, early stopping, and learning-rate reduction.
9. Select a decision threshold on validation data.
10. Evaluate once on the held-out test set.
11. Compare against a Random Forest one-hot baseline.
12. Save the model, preprocessing artifacts, metrics, visuals, and inference app.

## Original notebook improvements

The supplied notebook was a strong analysis prototype. The portfolio version corrects
and strengthens these areas:

- **Prevents encoder leakage:** the original fitted label encoders on combined train,
  validation, and test categories; retraining now fits on training data only.
- **Handles unseen categories:** the retraining encoder reserves OOV index 0. The
  included legacy model uses documented fallback categories so uploads never crash.
- **Separates reusable modules:** preprocessing, model building, evaluation,
  embedding analysis, and inference are moved into `src/`.
- **Adds deployment metadata:** feature order, thresholds, defaults, ranges, labels,
  and artifact compatibility are stored in JSON.
- **Uses validation thresholding:** threshold selection is separated from final test
  evaluation.
- **Adds production validation:** missing columns, numeric coercion, and unknown
  category counts are surfaced clearly.
- **Adds CI tests:** sample inference and unseen-category behavior are tested.
- **Adds a complete Streamlit workflow:** manual prediction, CSV upload, sample data,
  charts, warnings, and downloadable scored output.

## Visual results

### Training curves

![Training curves](outputs/training_curves.png)

### Classification evaluation

![Confusion matrix](outputs/confusion_matrix.png)

![ROC curve](outputs/roc_curve.png)

![Precision recall curve](outputs/precision_recall_curve.png)

### Baseline comparison

![Model comparison](outputs/model_comparison.png)

### Embedding and feature analysis

![Embedding visualization](outputs/embedding_visualization.png)

![Permutation importance](outputs/permutation_importance.png)

## Streamlit demo

The app supports:

- manual single-record prediction;
- sample data or uploaded CSV batch scoring;
- input preview and required-feature display;
- probability, class, and business bucket output;
- unknown-category fallback warnings;
- prediction counts and probability distribution;
- downloadable scored CSV;
- model comparison and embedding insight visuals.

Run locally:

```bash
cd 10-tabular-deep-learning-with-embeddings
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

macOS/Linux activation:

```bash
source .venv/bin/activate
```

## Retrain the model

```bash
python train_model.py
```

Retraining creates a new model with a dedicated OOV category index and writes updated
artifacts to `models/` and `outputs/`.

## Folder structure

```text
10-tabular-deep-learning-with-embeddings/
├── .streamlit/config.toml
├── app/
│   ├── requirements.txt
│   └── streamlit_app.py
├── data/
│   ├── README_data.md
│   └── sample_input.csv
├── images/
│   └── README.md
├── models/
│   ├── feature_metadata.json
│   ├── label_encoders.pkl
│   ├── numeric_imputer.pkl
│   ├── numeric_scaler.pkl
│   └── tabular_embedding_ann.keras
├── notebooks/
│   ├── tabular_deep_learning_with_embeddings.ipynb
│   └── original_experiment_notebook.ipynb
├── outputs/
│   ├── model_metrics.json
│   ├── final_metrics_summary.csv
│   ├── training_history.csv
│   └── evaluation and embedding visuals
├── src/
│   ├── config.py
│   ├── data_generation.py
│   ├── data_preprocessing.py
│   ├── embedding_analysis.py
│   ├── embedding_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   └── prediction_pipeline.py
├── tests/test_preprocessing.py
├── .gitignore
├── README.md
├── README_HOSTING.md
├── PROJECT_REVIEW.md
├── REPOSITORY_INTEGRATION.md
├── FILE_MANIFEST.csv
├── requirements.txt
├── run_local.bat
└── train_model.py
```


## Recommended screenshots

After the app is running, save these files under `images/`:

1. `01_app_overview.png` — title, model explanation, and threshold control.
2. `02_single_prediction.png` — completed manual input and probability result.
3. `03_batch_prediction.png` — input preview, scored summary, and charts.
4. `04_download_scored_csv.png` — scored table and download workflow.
5. `05_model_insights.png` — baseline comparison and embedding analysis.

Use the strongest two or three screenshots in this README after deployment. The
pre-generated analytical visuals under `outputs/` can remain visible even before the
live-demo screenshots are added.

## Deployment

Streamlit Community Cloud is recommended for the first hosted version. Use:

```text
10-tabular-deep-learning-with-embeddings/app/streamlit_app.py
```

as the app entrypoint and select Python 3.12. See [README_HOSTING.md](README_HOSTING.md)
for the full deployment checklist.

## Skills demonstrated

- Keras functional API and multi-input neural networks
- categorical embeddings for mixed tabular data
- binary classification and class-imbalance handling
- leakage-aware preprocessing and validation design
- threshold analysis and probability-based decision support
- baseline comparison and error-oriented evaluation
- embedding extraction, PCA, and similarity analysis
- reusable inference pipelines and model artifact management
- Streamlit deployment and batch scoring
- GitHub Actions continuous integration

## Portfolio positioning

**One-line description:**  
Built a multi-input Keras ANN that learns categorical embeddings, combines them with
scaled numerical features, and deploys probability-based batch scoring through
Streamlit.

**Pinned repository description:**  
Advanced tabular deep learning project using categorical embeddings, leakage-aware
preprocessing, baseline comparison, embedding analysis, and a hosted Streamlit demo.

This project supports a transition from Quality Data Science into broader ML and AI
roles by demonstrating end-to-end model engineering rather than notebook-only
experimentation: reproducible data, modular training, robust inference, evaluation,
visual interpretation, CI, and deployment.

## Future improvements

- retrain on a real, permissioned high-cardinality dataset;
- compare against CatBoost and modern tabular architectures such as FT-Transformer;
- add SHAP or integrated gradients for local explanations;
- calibrate probabilities with Platt scaling or isotonic regression;
- add model monitoring for drift, OOV rates, and prediction stability;
- use MLflow or Weights & Biases for experiment tracking;
- containerize the app for Hugging Face Spaces or cloud deployment.
