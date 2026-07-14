# Project Overview

## Business Problem

Customer churn affects revenue, retention costs, and long-term customer value.
A churn model can help a retention team prioritize customers for review before
they leave.

## Analytical Objective

Estimate the probability of churn from customer profile and account
characteristics using an artificial neural network.

## Intended User

A retention analyst or customer-experience team using the score as one input
for prioritizing outreach.

## Portfolio Value

This project demonstrates:

- Binary classification with an ANN
- Categorical encoding and feature scaling
- Hyperparameter experimentation
- Class-imbalance awareness
- Probability-based inference
- Interactive Streamlit deployment
- Batch prediction
- Input validation
- Reproducible training code
- Testing and CI structure

## Original Model Audit

The original notebook produced:

- Test accuracy: 85.65%
- Churn precision: 70.08%
- Churn recall: 47.07%
- Churn F1: 56.32%
- ROC-AUC: 85.62%
- PR-AUC: 67.63%

The most important limitation is churn recall. At the 0.50 threshold, the model
missed 208 of 393 churners in the original test split.

The original final training also used the test split as validation data for
early stopping. The improved training script corrects this by creating separate
stratified train, validation, and test sets.
