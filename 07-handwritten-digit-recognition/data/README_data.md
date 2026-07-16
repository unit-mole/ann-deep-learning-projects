# Data Guide

This project uses the **MNIST handwritten digit dataset** through:

```python
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

MNIST contains 60,000 training images and 10,000 test images. Each sample is a 28×28 grayscale digit labeled from 0 through 9.

## Why the raw dataset is not committed

TensorFlow/Keras downloads and caches MNIST automatically on first use. Keeping the full dataset outside Git avoids unnecessary repository size and duplicate distribution.

## Included sample images

`sample_digits/` contains a small set of MNIST examples extracted from the executed project notebook. They are used by the Streamlit demo and are safe to keep in Git.

## Expected custom image format

The app accepts PNG, JPG, JPEG, or BMP files. Custom images may be black-on-white or white-on-black. The preprocessing pipeline automatically converts them to a centered 28×28 white-on-black representation and normalizes pixels to `[0, 1]`.

Dataset reference: https://keras.io/api/datasets/mnist/
