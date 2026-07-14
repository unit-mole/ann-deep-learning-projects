# Dataset Setup

## Full Dataset

This project uses the public Credit Card Fraud Detection dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading the archive, place the full CSV at:

```text
ANN/Credit_Card_Fraud_Detection/data/creditcard.csv
```

The supplied local file audited for this project had:

| Property | Value |
|---|---:|
| Rows | 284,807 |
| Columns | 31 |
| Model features | 30 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud rate | 0.1727% |
| Missing values | 0 |
| Approximate file size | 151 MB |

## Why the Full CSV Is Not Included

The full dataset is excluded from the GitHub-ready package because:

- it is much larger than the files needed to demonstrate inference;
- dataset licensing and redistribution terms should be respected;
- large source files make repositories slower to clone and deploy;
- training data should remain separate from deployable application artifacts.

The project `.gitignore` excludes `data/*.csv` except for
`data/sample_input.csv`.

## Sample Input

`sample_input.csv` contains 50 rows with the complete feature schema:

```text
Time, V1, V2, ..., V28, Amount
```

It also includes the optional `Class` column so the Streamlit app can display
demonstration evaluation metrics.

The sample is deliberately curated to include:

- 40 legitimate rows
- 10 fraudulent rows

This is **not representative** of the source fraud prevalence. It exists only
to make both prediction classes visible during a portfolio demonstration.

## Production Input Schema

Required numeric columns:

```text
Time
V1 through V28
Amount
```

Optional columns:

```text
Class
transaction_id
```

Extra columns are preserved in the prediction output, but the model scores only
the required 30 features.

## Safety Notes

- Do not upload real customer card numbers, names, account identifiers, or other
  sensitive financial data to a public demo.
- The public dataset is anonymized.
- The app performs technical input validation but is not a production security
  boundary.
