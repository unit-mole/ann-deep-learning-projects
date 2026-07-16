# Model Card

## Model details

- **Name:** MNIST Dense ANN
- **Task:** Single-image handwritten digit classification
- **Classes:** 0–9
- **Framework:** TensorFlow/Keras
- **Input:** One normalized 28×28 grayscale image flattened to 784 features
- **Output:** Ten softmax class probabilities
- **Architecture:** 784 → 384 ReLU → Dropout 0.20 → 192 ReLU → Dropout 0.20 → 10 Softmax
- **Parameters:** 377,290

## Intended use

Educational and portfolio demonstration of ANN-based image classification and deployment. Suitable for clean, isolated handwritten digits that resemble MNIST.

## Out-of-scope use

Not designed for identity verification, high-stakes document processing, multi-digit OCR, or decisions affecting people. Do not treat softmax confidence as calibrated certainty.

## Performance

- Test accuracy: 98.24%
- Macro F1: 98.23%
- Correct predictions: 9,824 of 10,000
- Lowest class recall in the recorded test run: digit 8 at 97.23%

## Data

MNIST contains 60,000 training and 10,000 test 28×28 grayscale images. The training portion is split into 54,000 training and 6,000 validation images.

## Limitations and risks

Performance can degrade when inputs contain shadows, texture, multiple characters, weak contrast, extreme rotation, or handwriting unlike MNIST. The app mitigates common polarity, scaling, and centering issues but does not eliminate domain shift.
