# ANN Deep Learning Projects

A structured portfolio of artificial neural network projects covering tabular classification, regression, forecasting, computer vision, optimization, and mixed-output prediction.

**Portfolio status:** 9 completed and deployed projects · 2 planned projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)

---

## Portfolio Objective

This repository demonstrates how artificial neural networks can be applied to practical business and analytical problems. Each completed project is developed as an end-to-end case study containing:

- a clearly defined business problem;
- reproducible data preparation and feature engineering;
- ANN model development and task-appropriate evaluation;
- saved inference artifacts and modular prediction code;
- an interactive Streamlit demonstration;
- automated tests and GitHub Actions CI;
- deployment guidance;
- an honest discussion of limitations and future improvements.

The portfolio is designed to demonstrate skills relevant to Data Science, Machine Learning, Applied AI, Data Analytics, Quality Analytics, and Analytics Engineering roles.

---

## Project Roadmap

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
| 10 | Salary Prediction | Regression | Planned |
| 11 | Tabular Deep Learning with Embeddings | Tabular deep learning | Planned |

---

## Latest Completed Project

### [09 — Multi-Output Prediction System](09-multi-output-prediction-system/)

A shared multi-head ANN that predicts customer churn probability, customer lifetime value, and engagement score in one inference workflow.

The project demonstrates:

- mixed-output learning with classification and regression heads;
- shared representation learning;
- leakage-safe preprocessing;
- validation-based threshold selection;
- manual and batch scoring;
- downloadable predictions;
- output-specific evaluation;
- GitHub Actions CI and Streamlit deployment.

[Open the live application](https://ann-deep-learning-projects-5mvtt4spt9hwj28ytb8gze.streamlit.app/)

---

## Repository Convention

The repository is organized as a monorepo. Each completed project generally follows this structure:

```text
ann-deep-learning-projects/
├── .github/
│   └── workflows/
│       └── project-specific-ci.yml
│
├── project-folder/
│   ├── app/
│   │   └── streamlit_app.py
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
│   ├── requirements-ci.txt
│   └── streamlit_app.py
│
├── LICENSE
└── README.md
```

The exact contents vary by project, but the core goals remain consistent: reproducibility, modular code, deployable inference, testing, documentation, and responsible model communication.

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
- documenting model limitations transparently.

---

## Core Skills Demonstrated

`Artificial Neural Networks` · `TensorFlow` · `Keras` · `scikit-learn` · `Feature Engineering` · `Classification` · `Regression` · `Forecasting` · `Computer Vision` · `Multi-task Learning` · `Model Evaluation` · `Streamlit` · `Testing` · `CI/CD` · `Business Translation`

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics
