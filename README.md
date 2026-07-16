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

## Latest Completed Project

### [10 — Tabular Deep Learning with Embeddings](10-tabular-deep-learning-with-embeddings/)

A multi-input Keras neural network that learns dense representations for categorical business variables, combines them with scaled numerical features, and produces a positive-outcome propensity score.

The project demonstrates:

- separate categorical and numerical model inputs;
- trainable embedding layers for seven categorical features;
- embedding and numerical feature concatenation;
- leakage-aware preprocessing and saved inference artifacts;
- safe handling of previously unseen categories;
- class-weighted binary classification;
- validation-based threshold analysis;
- comparison with a Random Forest baseline;
- PCA-based embedding visualization and permutation importance;
- manual and batch prediction through Streamlit;
- automated tests and GitHub Actions CI.

[Open the live application](https://ann-deep-learning-projects-budbjucqqrtaar2bjin76u.streamlit.app/)

---

## Portfolio Coverage

The completed projects collectively demonstrate:

- binary and multi-class classification;
- regression and forecasting;
- imbalanced-learning workflows;
- probability scoring and threshold selection;
- multi-task and multi-output neural networks;
- customer analytics and risk-oriented modeling;
- computer vision;
- constrained optimization;
- categorical embeddings for structured tabular data;
- single-record and batch inference;
- interactive application deployment;
- automated testing and continuous integration.

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

The exact contents vary by project, but the core goals remain consistent: reproducibility, modular code, deployable inference, automated testing, clear documentation, and responsible model communication.

---

## Portfolio Positioning

These projects are designed to show more than notebook-based experimentation. The portfolio emphasizes:

- translating business problems into machine-learning solutions;
- selecting evaluation metrics appropriate to the target and class distribution;
- preventing data leakage and fitting preprocessing on training data only;
- building reusable prediction pipelines;
- combining technical outputs with business-friendly interpretations;
- supporting manual and batch inference;
- deploying interactive applications;
- testing important code paths through CI;
- comparing ANN models with appropriate baselines;
- documenting model limitations transparently.

The portfolio supports a transition from Quality Data Science into broader Data Science, Machine Learning, Applied AI, Analytics, and Analytics Engineering roles by demonstrating complete project delivery from problem framing through deployment.

---

## Core Skills Demonstrated

`Artificial Neural Networks` · `TensorFlow` · `Keras` · `scikit-learn` · `Feature Engineering` · `Categorical Embeddings` · `Classification` · `Regression` · `Forecasting` · `Computer Vision` · `Multi-task Learning` · `Multi-output Learning` · `Class-Imbalance Handling` · `Threshold Analysis` · `Model Evaluation` · `Streamlit` · `Testing` · `GitHub Actions` · `CI/CD` · `Business Translation`

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics
