# Data Guide

## Included Files

- `raw/Churn_Modelling.csv`: dataset supplied with the original project
- `sample/batch_prediction_sample.csv`: small demo file for batch scoring

## Target

`Exited`

- `0`: customer retained
- `1`: customer churned

## Prediction Features

| Feature | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | France, Germany, or Spain |
| Gender | Female or Male |
| Age | Customer age |
| Tenure | Number of years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is active |
| EstimatedSalary | Estimated annual salary |

## Excluded Identifiers

`RowNumber`, `CustomerId`, and `Surname` are excluded from model training.

## Before Public Release

The dataset source and redistribution license were not documented in the
original files supplied for this conversion. Before publishing the full raw
CSV publicly:

1. Confirm the original dataset source.
2. Add the source link and license.
3. Confirm that redistribution is permitted.
4. Otherwise, keep the raw CSV out of GitHub and publish only a synthetic sample.
