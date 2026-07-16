# Dataset Guide

## Included dataset

`house_prices.csv` contains the **California Housing** regression dataset used by
the uploaded notebook and saved ANN artifacts.

- Rows: 20,640
- Predictors: 8 numeric block-group features
- Target: `SalePrice`
- Target unit: hundreds of thousands of US dollars
- Example: `SalePrice = 4.25` represents approximately `$425,000`

## Important scope limitation

This dataset describes California census block groups. `AveRooms`,
`AveBedrms`, and `AveOccup` are area-level averages, while `SalePrice` is the
median house value for the block group. It is not an individual home-sales
dataset.

## Required input columns

```text
MedInc
HouseAge
AveRooms
AveBedrms
Population
AveOccup
Latitude
Longitude
```

`sample_input.csv` contains valid feature-only rows for the Streamlit batch demo.

## Data safety

The included CSV is approximately 1.9 MB and contains no personal or proprietary
customer information. For future private datasets, place raw files in
`data/private/` and add that location to `.gitignore`.
