# Dataset notes

The included `diabetes.csv` is the small, public Pima Indians Diabetes benchmark dataset used by the original notebook. It contains 768 rows, eight numeric predictors, and the binary target `Outcome`.

## Expected columns

`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, and `Outcome`.

## Data-quality treatment

The raw file contains no explicit null values, but it uses zero as a likely missing-value marker in several medical fields. The training pipeline converts zero to missing for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`, then fits median imputation on the training partition only. Zero remains valid for `Pregnancies`.

The observed zero-marker counts are:

| Field | Count |
|---|---:|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

`sample_input.csv` contains only the eight model inputs and can be uploaded directly in the Streamlit app.

## Safety

Do not replace this file with private patient data in a public repository. Use de-identified or synthetic data for portfolio demonstrations.
