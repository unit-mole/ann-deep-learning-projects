# Portfolio Positioning

## One-Line Project Description

Built an ANN-based fraud-risk scoring pipeline for highly imbalanced
transactions, with class weighting, PR-AUC evaluation, threshold controls, and
a deployable Streamlit batch-prediction application.

## GitHub Repository Description

```text
ANN portfolio project for credit-card fraud risk scoring using TensorFlow,
class-imbalance handling, probability-based predictions, threshold analysis,
and a live Streamlit demo.
```

Keep the GitHub repository description below approximately 160 characters when
possible:

```text
ANN fraud detection with TensorFlow, imbalance-aware evaluation, threshold
tuning, probability scoring, and a Streamlit demo.
```

## Suggested Pinned Repository Summary

```text
Credit Card Fraud Detection using ANN

End-to-end deep-learning project for a 0.17%-fraud transaction dataset.
Includes stratified preprocessing, class weighting, PR-AUC and ROC-AUC
evaluation, threshold controls, saved inference artifacts, and a hosted
Streamlit batch-scoring app.
```

## Resume Bullet Options

### Technical version

```text
Developed an ANN-based credit-card fraud classifier using TensorFlow/Keras for
284K highly imbalanced transactions; implemented stratified preprocessing,
class-weighted training, PR-AUC/ROC-AUC evaluation, probability scoring, and a
deployable Streamlit inference workflow.
```

### Outcome-oriented version

```text
Built an end-to-end fraud-risk scoring system that converts ANN probabilities
into configurable investigation alerts, with reproducible preprocessing,
imbalance-aware metrics, artifact persistence, and an interactive web demo.
```

### Concise version

```text
Built and deployed an ANN fraud-detection pipeline with class-imbalance
handling, threshold tuning, and batch probability scoring in Streamlit.
```

## Skills Demonstrated

### Data Science

- imbalanced binary classification;
- train/validation/test design;
- leakage-aware preprocessing;
- probability-based modeling;
- metric interpretation;
- error analysis.

### Machine Learning and AI

- feed-forward neural networks;
- TensorFlow and Keras;
- activation functions and dropout;
- binary cross-entropy;
- class weighting;
- early stopping;
- threshold optimization.

### Analytics Engineering

- modular Python package structure;
- feature contracts;
- validated inference inputs;
- saved preprocessing artifacts;
- reproducible configuration;
- automated tests.

### Business and Risk Analytics

- false-positive and false-negative trade-offs;
- fraud review prioritization;
- risk-score interpretation;
- operational threshold controls;
- model governance considerations.

### Deployment

- Streamlit application development;
- GitHub-based deployment;
- dependency management;
- model loading and batch scoring;
- downloadable outputs.

## Recommended Screenshots

1. **Repository hero screenshot**  
   README title, badges, business problem, and key metrics.

2. **Streamlit landing screen**  
   App title, disclaimer, input mode, and threshold control.

3. **Batch prediction summary**  
   Transactions scored, SAFE count, FRAUD/RISK count, and fraud rate.

4. **Highest-risk transactions**  
   Probability progress bars and risk labels.

5. **Model evaluation**  
   Confusion matrix and precision-recall curve side by side.

6. **Threshold demonstration**  
   The same sample scored at two thresholds to show operational trade-offs.

## Recommended Output Images

Keep these committed in `outputs/`:

- class distribution;
- ANN architecture;
- confusion matrix;
- ROC curve;
- precision-recall curve.

Add these after running the application:

- app screenshot;
- batch-prediction screenshot;
- threshold-comparison screenshot.

## How This Supports a Career Transition

This project connects your current Quality Data Scientist experience with
broader Data Science and AI roles in several credible ways:

- **Rare-event detection:** fraud events resemble low-frequency quality failures
  where accuracy can hide poor minority-class performance.
- **Risk prioritization:** probability scores support triage, investigation, and
  resource allocation, similar to prioritizing high-risk quality cases.
- **False-alert management:** balancing fraud recall against customer friction
  mirrors balancing defect detection against unnecessary escalations.
- **End-to-end ownership:** the project goes beyond a notebook into reusable
  code, artifacts, testing, documentation, and deployment.
- **Business communication:** the README explains not only model metrics but why
  those metrics affect operational decisions.
- **Transferable architecture:** the same pipeline pattern applies to quality
  escapes, warranty claims, supplier risk, anomaly detection, and predictive
  maintenance.

## Interview Talking Points

Be ready to explain:

1. Why 99.8% accuracy could still represent a useless fraud model.
2. Why PR-AUC is especially valuable when fraud prevalence is very low.
3. Why the scaler must be fitted only on training data.
4. Why class weights were preferred over SMOTE for the default pipeline.
5. How lowering the threshold affects recall and false positives.
6. Why validation data selects the threshold and test data measures final
   generalization.
7. Why a real bank would require monitoring, governance, security, and
   human-review workflows beyond this demo.
8. What you would compare against the ANN before production deployment.
