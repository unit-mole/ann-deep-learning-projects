# Dataset notes

The included `diabetes.csv` is the small public **Pima Indians Diabetes** benchmark dataset used by the original notebook. It contains 768 rows, eight numeric predictors, and the binary target `Outcome`.

## Dataset provenance and attribution

The CSV in this repository was supplied with the original project notebook. The exact website or mirror from which this particular copy was downloaded was not recorded, so this repository does not claim a specific download source.

Historical attribution for the benchmark dataset is commonly given to the **National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)**. The dataset is associated with the following early neural-network study:

- J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, “Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus,” 1988.  
  Research article: [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/)

A commonly used public reference mirror is available through Kaggle:

- [Pima Indians Diabetes Database — Kaggle/UCI ML mirror](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The Kaggle page is included as a reference mirror only; it is not asserted to be the exact source of the repository copy.

## Usage and licensing note

The project’s MIT license applies to the project code and documentation. It does **not** independently establish ownership or licensing rights for the dataset.

Before redistributing the dataset outside this educational repository or using it for commercial work, verify the terms and attribution requirements of the source from which the data is obtained. This project uses the data only as a public benchmark for educational and portfolio demonstration.

## Expected columns

`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, and `Outcome`.

`sample_input.csv` contains only the eight model inputs and can be uploaded directly in the Streamlit app.

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

For uploaded batch files, the inference pipeline:

- requires all eight feature columns;
- allows genuine missing values for median imputation;
- rejects unexpected text instead of silently converting it to missing;
- rejects infinite and negative measurements;
- converts documented zero markers to missing only for the five fields listed above.

## Safety and responsible use

- Do not replace the included benchmark file with private patient data in a public repository.
- Use de-identified or synthetic data for portfolio demonstrations.
- The dataset has limited demographic and historical scope and is not representative of all populations.
- The model and dataset have not been clinically validated for diagnostic use.
