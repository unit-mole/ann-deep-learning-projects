# ANN Deep Learning Projects

A structured portfolio of ten completed artificial neural network projects covering tabular classification, imbalanced learning, probability scoring, forecasting, optimization, computer vision, multi-output learning, and categorical embeddings.

**Portfolio status:** 10 completed and deployed projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)

---

## Portfolio Objective

This repository demonstrates how artificial neural networks can be applied to practical business and analytical problems. Each project is developed as an end-to-end case study containing:

- a clearly defined business problem;
- reproducible data preparation and feature engineering;
- ANN model development and task-appropriate evaluation;
- saved inference artifacts and modular prediction code;
- an interactive Streamlit demonstration;
- automated tests and GitHub Actions CI;
- deployment guidance;
- an honest discussion of limitations and future improvements.

The portfolio is designed to demonstrate skills relevant to Data Science, Machine Learning, Applied AI, Data Analytics, Quality Analytics, Business Intelligence, and Analytics Engineering roles.

---

## Completed Projects

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Churn Classification](01-churn-classification/) | Binary classification | [Live Demo](https://churn-risk-ann.streamlit.app) |
| 2 | [Credit Card Fraud Detection](02-credit-card-fraud-detection/) | Imbalanced binary classification | [Live Demo](https://ann-deep-learning-projects-dqnj5rpwbpmuxtd2tcm5mh.streamlit.app/) |
| 3 | [Credit Risk Probability Scoring](03-credit-risk-probability-scoring/) | Risk classification and probability scoring | [Live Demo](https://ann-deep-learning-projects-9p9vupmu9kbk5462v6hbkb.streamlit.app/) |
| 4 | [Customer Lifetime Value Forecasting](04-customer-lifetime-value-forecasting/) | Multi-task regression and retention prediction | [Live Demo](https://ann-deep-learning-projects-u4gymvvpwuaowqnmkjq3wa.streamlit.app/) |
| 5 | [Diabetes Risk Screening](05-diabetes-prediction/) | Healthcare risk classification and probability scoring | [Live Demo](https://ann-deep-learning-projects-bczyq9q5aa8eqbvqskqyar.streamlit.app/) |
| 6 | [Dynamic Pricing Optimization System](06-dynamic-pricing-optimization/) | Demand forecasting and constrained price optimization | [Live Demo](https://ann-deep-learning-projects-tgcmwtdfyxorbrexrmbcin.streamlit.app/) |
| 7 | [Handwritten Digit Recognition](07-handwritten-digit-recognition/) | Multi-class computer vision | [Live Demo](https://ann-deep-learning-projects-gsnhfzexxframenzenm5rx.streamlit.app/) |
| 8 | [House Price Prediction](08-house-price-prediction/) | ANN-based tabular regression and price estimation | [Live Demo](https://ann-deep-learning-projects-satmbakncxmlo2mmct5gvu.streamlit.app/) |
| 9 | [Multi-Output Prediction System](09-multi-output-prediction-system/) | Mixed-output ANN for churn, customer value, and engagement prediction | [Live Demo](https://ann-deep-learning-projects-5mvtt4spt9hwj28ytb8gze.streamlit.app/) |
| 10 | [Tabular Deep Learning with Embeddings](10-tabular-deep-learning-with-embeddings/) | Binary classification using categorical embeddings and numerical features | [Live Demo](https://ann-deep-learning-projects-budbjucqqrtaar2bjin76u.streamlit.app/) |

---

## What the Portfolio Covers

The ten projects are intentionally varied so that the repository demonstrates more than one type of neural-network problem.

### Customer and Business Analytics

- **Churn Classification** identifies customers with an elevated likelihood of leaving.
- **Customer Lifetime Value Forecasting** predicts future customer value and retention probability.
- **Multi-Output Prediction System** produces churn, value, and engagement predictions through one shared neural network.
- **Dynamic Pricing Optimization System** combines demand prediction with business-constrained price recommendations.

These projects demonstrate how predictive models can support retention, prioritization, customer segmentation, revenue planning, and operational decision-making.

### Risk, Screening, and Anomaly Detection

- **Credit Card Fraud Detection** focuses on severe class imbalance and minority-event detection.
- **Credit Risk Probability Scoring** generates both a predicted risk class and an interpretable probability score.
- **Diabetes Risk Screening** demonstrates probability-based health-risk classification while clearly separating a portfolio prototype from clinical diagnosis.

These projects emphasize recall, precision, F1-score, ROC-AUC, PR-AUC, threshold selection, probability interpretation, and responsible communication of risk-oriented model outputs.

### Regression and Forecasting

- **House Price Prediction** estimates continuous property values from structured attributes.
- **Customer Lifetime Value Forecasting** predicts future monetary value.
- **Dynamic Pricing Optimization System** forecasts demand and evaluates candidate pricing decisions.

These projects demonstrate numerical preprocessing, continuous-output ANN design, MAE, RMSE, R², residual analysis, and business interpretation of regression errors.

### Computer Vision

- **Handwritten Digit Recognition** applies convolutional neural-network concepts to multi-class image classification.

This project expands the portfolio beyond tabular data and demonstrates image preprocessing, class probabilities, confusion-matrix analysis, and interactive inference.

### Advanced Tabular Deep Learning

- **Tabular Deep Learning with Embeddings** learns dense representations of categorical features and combines them with scaled numerical inputs.
- **Multi-Output Prediction System** uses shared representation learning across classification and regression targets.
- **Customer Lifetime Value Forecasting** applies multi-task learning with categorical embeddings and separate prediction heads.

These projects demonstrate more advanced ANN architectures than a standard sequential network, including the Keras Functional API, multiple inputs, embeddings, shared layers, multiple outputs, and reusable inference pipelines.

---

## What the Repository Demonstrates

### End-to-End Machine Learning Delivery

Every project is structured to move beyond notebook-only experimentation. The repository demonstrates:

- business-problem definition;
- reproducible data preparation;
- feature engineering;
- training, validation, and test separation;
- model evaluation;
- saved preprocessing and model artifacts;
- reusable prediction pipelines;
- manual and batch inference;
- downloadable outputs;
- local execution;
- cloud deployment.

### Model Evaluation Based on the Actual Problem

The projects use evaluation metrics that match the task rather than relying on accuracy alone.

Examples include:

- precision, recall, F1, ROC-AUC, and PR-AUC for classification;
- MAE, RMSE, R², and residual analysis for regression;
- output-specific metrics for multi-task models;
- confusion matrices and probability distributions;
- threshold analysis for imbalanced and risk-oriented decisions;
- baseline comparisons where they add meaningful context.

### Reliable and Reusable Engineering

The repository includes practices required for dependable inference:

- preprocessing fitted on training data only;
- consistent feature order between training and prediction;
- saved scalers, imputers, encoders, and metadata;
- safe handling of missing values and unseen categories;
- modular source files rather than notebook-only logic;
- automated tests for important prediction paths;
- project-specific GitHub Actions workflows;
- Streamlit deployment from the main repository branch.

### Business Translation

The applications do not stop at raw model outputs. Depending on the project, they provide:

- risk levels;
- prediction probabilities;
- customer segments;
- pricing recommendations;
- value estimates;
- business interpretations;
- batch summaries;
- downloadable scored datasets.

This demonstrates the ability to translate technical predictions into information that can be understood and used by business stakeholders.

### Responsible Model Communication

Each project documents its intended scope and limitations. Synthetic or sample data is identified clearly, and the applications avoid presenting portfolio models as production-ready financial, medical, or operational decision systems.

---

## Repository Convention

The repository is organized as a monorepo. Each project generally follows this structure:

```text
ann-deep-learning-projects/
├── .github/
│   └── workflows/
│       └── project-specific-ci.yml
│
├── project-folder/
│   ├── app/
│   │   ├── streamlit_app.py
│   │   └── requirements.txt
│   ├── data/
│   │   ├── sample_input.csv
│   │   └── README_data.md
│   ├── images/
│   ├── models/
│   ├── notebooks/
│   ├── outputs/
│   ├── src/
│   ├── tests/
│   ├── .gitignore
│   ├── README.md
│   ├── README_HOSTING.md
│   ├── requirements.txt
│   └── supporting project files
│
├── LICENSE
└── README.md
```

The exact files vary by project, but the standards remain consistent:

- reproducible workflows;
- modular code;
- deployable inference;
- automated validation;
- clear documentation;
- safe repository practices;
- transparent model limitations.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Binary classification | Churn, fraud, credit risk, diabetes, tabular embeddings |
| Multi-class classification | Handwritten digit recognition |
| Regression | House price prediction, customer value forecasting |
| Multi-task and multi-output learning | Customer lifetime value and multi-output prediction |
| Imbalanced learning | Fraud detection, risk scoring, positive-outcome propensity |
| Categorical embeddings | CLV forecasting and tabular deep learning |
| Computer vision | Handwritten digit recognition |
| Optimization | Dynamic pricing |
| Probability scoring | Churn, credit risk, diabetes, fraud, tabular embeddings |
| Manual inference | Interactive Streamlit input forms |
| Batch inference | CSV upload, sample scoring, downloadable outputs |
| Model deployment | Ten Streamlit Community Cloud applications |
| Testing and CI/CD | pytest and project-specific GitHub Actions workflows |

---

## Core Skills Demonstrated

`Artificial Neural Networks` · `TensorFlow` · `Keras` · `Keras Functional API` · `scikit-learn` · `pandas` · `Feature Engineering` · `Categorical Embeddings` · `Classification` · `Regression` · `Forecasting` · `Computer Vision` · `Multi-task Learning` · `Multi-output Learning` · `Class-Imbalance Handling` · `Threshold Analysis` · `Probability Scoring` · `Model Evaluation` · `Streamlit` · `Testing` · `GitHub Actions` · `CI/CD` · `Business Translation`

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics
