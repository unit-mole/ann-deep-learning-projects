# Handwritten Digit Recognition with an Artificial Neural Network

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Open_App-FF4B4B?logo=streamlit&logoColor=white)](https://ann-deep-learning-projects-gsnhfzexxframenzenm5rx.streamlit.app/)
[![CI](https://github.com/unit-mole/ann-deep-learning-projects/actions/workflows/handwritten-digit-recognition-ci.yml/badge.svg)](https://github.com/unit-mole/ann-deep-learning-projects/actions/workflows/handwritten-digit-recognition-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deployment-ready computer-vision portfolio project that classifies handwritten digits from 0 to 9 using a tuned dense artificial neural network, MNIST-aligned image preprocessing, and an interactive Streamlit interface.

> **Live application:** [Launch the Handwritten Digit Recognition demo](https://ann-deep-learning-projects-gsnhfzexxframenzenm5rx.streamlit.app/)

## Project Snapshot

| Item | Result |
|---|---:|
| Dataset | MNIST |
| Training / validation / test | 54,000 / 6,000 / 10,000 |
| Model | Dense ANN with dropout |
| Test accuracy | **98.24%** |
| Macro F1 | **98.23%** |
| Correct test predictions | **9,824 / 10,000** |
| Misclassified test images | **176** |
| Trainable parameters | **377,290** |
| Deployment | **Streamlit Community Cloud** |

## Application Preview

The deployed interface supports built-in MNIST examples and user-uploaded handwritten digit images. It displays the complete preprocessing flow, ranked predictions, class probabilities, and downloadable results.

<table>
  <tr>
    <td width="50%" align="center">
      <img src="images/01_app_home.png" alt="Handwritten Digit Recognition application home page" width="100%">
      <br><strong>Application home and model summary</strong>
    </td>
    <td width="50%" align="center">
      <img src="images/02_sample_digit_7_preprocessing.png" alt="Sample digit preprocessing pipeline" width="100%">
      <br><strong>Sample selection and preprocessing</strong>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="images/03_sample_digit_7_prediction.png" alt="Digit 7 prediction result" width="100%">
      <br><strong>Predicted digit, confidence, and runner-up</strong>
    </td>
    <td width="50%" align="center">
      <img src="images/04_sample_digit_7_probability_distribution.png" alt="Class probability distribution" width="100%">
      <br><strong>Probability distribution across digits 0–9</strong>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="images/06_uploaded_digit_preprocessing.png" alt="Uploaded digit preprocessing" width="100%">
      <br><strong>User-uploaded image preprocessing</strong>
    </td>
    <td width="50%" align="center">
      <img src="images/07_uploaded_digit_prediction.png" alt="Uploaded digit prediction result" width="100%">
      <br><strong>Prediction for a user-uploaded digit</strong>
    </td>
  </tr>
</table>

<details>
<summary><strong>Additional interface view: probability table and CSV download</strong></summary>

<br>

![Probability table and download option](images/05_probability_table_and_download.png)

</details>

## Problem Statement

Given an image containing one handwritten digit, predict which class from 0 through 9 it represents and return:

- the predicted digit;
- prediction confidence;
- the second-highest prediction;
- probabilities for all ten classes.

Handwritten digit recognition is a foundational computer-vision problem with applications in optical character recognition, form processing, bank-check interpretation, postal automation, and document digitization.

## Dataset

The project uses MNIST, which contains 60,000 training images and 10,000 test images. Every image is grayscale and 28×28 pixels. The dataset is downloaded through the Keras dataset loader and is not committed to the repository.

The original 60,000-image training set is divided using a stratified 90/10 split:

- 54,000 training images;
- 6,000 validation images;
- 10,000 official test images.

See [`data/README_data.md`](data/README_data.md) for the data-handling policy.

## End-to-End Workflow

```text
MNIST / Uploaded Image
          │
          ▼
Grayscale + automatic inversion
          │
          ▼
Foreground crop + aspect-preserving resize
          │
          ▼
Centered 28×28 image + normalization
          │
          ▼
Flatten to 784 pixel features
          │
          ▼
Dense ANN: 384 → 192 → 10
          │
          ▼
Softmax probabilities + ranked prediction
```

## Image Preprocessing

### Training images

1. Load 28×28 grayscale MNIST images.
2. Convert pixels from integers in `[0, 255]` to `float32` values in `[0, 1]`.
3. Flatten each image into a 784-value vector.
4. One-hot encode labels for categorical cross-entropy.

### Uploaded images

Real uploads often use black ink on a white page, unlike MNIST's white digit on a black background. The reusable preprocessing pipeline therefore:

1. flattens transparent backgrounds onto white;
2. converts the image to grayscale;
3. detects background polarity and inverts when needed;
4. finds and crops the foreground digit;
5. resizes it within a 20×20 content box while preserving aspect ratio;
6. places the result on a 28×28 black canvas;
7. centers the digit using its intensity centroid;
8. normalizes the pixels and creates the model batch dimension.

This step is essential because the same digit can produce very different pixel vectors when its scale, centering, contrast, or polarity changes.

## ANN Architecture

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 784 normalized pixels | Flattened 28×28 image |
| Dense | 384 units, ReLU | Learn nonlinear pixel interactions |
| Dropout | 0.20 | Reduce overfitting |
| Dense | 192 units, ReLU | Compress learned features |
| Dropout | 0.20 | Additional regularization |
| Output | 10 units, Softmax | Probability for each digit |

Training uses Adam, categorical cross-entropy, early stopping, and learning-rate reduction. The selected search configuration uses a learning rate of `0.0005` and batch size of `256`.

## Evaluation Results

### Training performance

![Training accuracy and loss curves](outputs/accuracy_loss_curve.png)

### Confusion matrix

![Confusion matrix](outputs/confusion_matrix.png)

The model achieved **98.24% test accuracy**. Per-class recall ranges from approximately 97.23% for digit 8 to 99.30% for digit 1.

The most frequent observed confusions were:

- true 9 predicted as 4: 8 cases;
- true 8 predicted as 3: 7 cases;
- true 7 predicted as 2 or 9: 6 cases each;
- true 5 predicted as 3: 6 cases;
- true 4 predicted as 9: 6 cases.

These errors are visually plausible because handwritten digits can share strokes, loops, and incomplete closures.

### Correct predictions

![Sample correct predictions](outputs/sample_predictions.png)

### Misclassified examples

![Misclassified digit examples](outputs/misclassified_digits.png)

A confusion matrix reveals which classes are confused with one another rather than reducing performance to a single accuracy number. Misclassified-image review helps identify ambiguous handwriting, unusual stroke geometry, and potential preprocessing failures.

## Streamlit Demo

The live application supports:

- uploaded PNG, JPG, JPEG, or BMP images;
- built-in MNIST sample digits;
- original, grayscale, and final 28×28 previews;
- automatic black/white background inversion;
- predicted digit and confidence;
- runner-up prediction;
- probability table and interactive chart;
- CSV download of class probabilities;
- dynamic model metrics loaded from JSON.

**Open the application:**  
[https://ann-deep-learning-projects-gsnhfzexxframenzenm5rx.streamlit.app/](https://ann-deep-learning-projects-gsnhfzexxframenzenm5rx.streamlit.app/)

Drawing support was intentionally not added because upload and sample modes avoid an additional canvas dependency and are more reliable for Community Cloud deployment.

## Run Locally

Clone the repository and open the project directory:

```bat
git clone https://github.com/unit-mole/ann-deep-learning-projects.git
cd ann-deep-learning-projects\07-handwritten-digit-recognition
```

### Windows Command Prompt

The explicit virtual-environment interpreter is used below to avoid global Python and `PATH` conflicts:

```bat
python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
```

Open the local URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

### Run automated tests

On Windows:

```bat
".venv\Scripts\python.exe" -m pip install "pytest>=8,<10"
".venv\Scripts\python.exe" -m pytest tests -v
```

Expected result:

```text
6 passed
```

### Optional: retrain the ANN

Use the saved best configuration:

```bat
".venv\Scripts\python.exe" -m src.model_training --epochs 30
```

Repeat the original three-candidate search, then train a fresh final model:

```bat
".venv\Scripts\python.exe" -m src.model_training --tune --epochs 30
```

### Regenerate evaluation outputs

```bat
".venv\Scripts\python.exe" -m src.model_evaluation
```

MNIST downloads automatically the first time a training or evaluation command runs.

## Deployment

The application is deployed on Streamlit Community Cloud:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ann-deep-learning-projects-gsnhfzexxframenzenm5rx.streamlit.app/)

Use this monorepo deployment entrypoint:

```text
07-handwritten-digit-recognition/streamlit_app.py
```

Deployment configuration:

```text
Repository: unit-mole/ann-deep-learning-projects
Branch: main
Python: 3.13
Secrets: None required
```

The root-level `streamlit_app.py` is the intended Community Cloud entrypoint. The implementation remains modularized under `app/streamlit_app.py`.

Full deployment and troubleshooting instructions are available in [`README_HOSTING.md`](README_HOSTING.md).

## Repository Structure

```text
07-handwritten-digit-recognition/
├── .streamlit/
│   └── config.toml
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
├── data/
│   ├── sample_digits/
│   └── README_data.md
├── images/
│   ├── 01_app_home.png
│   ├── 02_sample_digit_7_preprocessing.png
│   ├── 03_sample_digit_7_prediction.png
│   ├── 04_sample_digit_7_probability_distribution.png
│   ├── 05_probability_table_and_download.png
│   ├── 06_uploaded_digit_preprocessing.png
│   ├── 07_uploaded_digit_prediction.png
│   └── README.md
├── models/
│   ├── best_params.json
│   └── digit_recognition_model.keras
├── notebooks/
│   └── handwritten_digit_recognition.ipynb
├── outputs/
│   ├── accuracy_loss_curve.png
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── misclassified_digits.png
│   ├── model_metrics.json
│   ├── per_class_accuracy.csv
│   ├── sample_predictions.png
│   ├── training_history.csv
│   └── training_history.json
├── src/
│   ├── data_preprocessing.py
│   ├── image_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   └── prediction_pipeline.py
├── tests/
├── .gitignore
├── FILE_MANIFEST.csv
├── IMPLEMENTATION_SUMMARY.md
├── LICENSE
├── MODEL_CARD.md
├── PORTFOLIO_COPY.md
├── README.md
├── README_HOSTING.md
├── requirements-dev.txt
├── requirements.txt
└── streamlit_app.py
```

The monorepo also includes the CI workflow at:

```text
.github/workflows/handwritten-digit-recognition-ci.yml
```

## Skills Demonstrated

- artificial neural networks and deep learning;
- multiclass image classification;
- TensorFlow/Keras training and serialization;
- image normalization and MNIST-style preprocessing;
- hyperparameter comparison and dropout regularization;
- confusion-matrix and per-class error analysis;
- modular Python application design;
- Streamlit model serving and visualization;
- testing, CI, reproducibility, and deployment documentation.

## Limitations

- A dense ANN discards explicit two-dimensional spatial structure when flattening images. A CNN is a logical benchmark for future comparison.
- MNIST is clean and standardized. Phone photographs, multiple digits, textured paper, shadows, and highly unusual handwriting create domain shift.
- Softmax confidence is not calibrated certainty. A high probability can still be wrong on out-of-distribution images.
- This project recognizes one isolated digit at a time; it is not a complete OCR pipeline.

## Future Improvements

- Train a CNN and compare accuracy, parameter count, and latency with the dense ANN.
- Add augmentation for rotation, translation, scale, blur, and brightness.
- Calibrate probabilities using temperature scaling.
- Add out-of-distribution and blank-image detection.
- Export to LiteRT/TensorFlow Lite for mobile or edge inference.
- Extend from isolated digits to multi-digit segmentation and OCR.

## Portfolio Positioning

This project demonstrates the transition from a notebook experiment to a reusable ML product: the original trained model is preserved, preprocessing is productionized, evaluation is documented, tests and CI are added, and inference is exposed through a deployed application. It complements quality and business analytics projects with image-based deep learning and model-serving experience.

## License

Released under the MIT License.
