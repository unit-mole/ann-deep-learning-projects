"""Preprocess arbitrary user images into MNIST-like ANN inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, ImageOps

ImageSource = Union[str, Path, BinaryIO, Image.Image]


@dataclass(frozen=True)
class PreprocessedDigit:
    """Artifacts created while converting an image to model-ready form."""

    original: Image.Image
    grayscale: Image.Image
    centered_28x28: Image.Image
    model_input: np.ndarray
    was_inverted: bool


def _open_image(source: ImageSource) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.copy()
    return Image.open(source)


def _flatten_transparency(image: Image.Image) -> Image.Image:
    """Place transparent images on white before grayscale conversion."""
    rgba = image.convert('RGBA')
    background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert('RGB')


def _border_mean(array: np.ndarray) -> float:
    top = array[0, :]
    bottom = array[-1, :]
    left = array[:, 0]
    right = array[:, -1]
    return float(np.concatenate([top, bottom, left, right]).mean())


def _translate(image: Image.Image, dx: int, dy: int) -> Image.Image:
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )


def preprocess_digit_image(
    source: ImageSource,
    *,
    auto_invert: bool = True,
    content_size: int = 20,
    foreground_threshold: int = 20,
) -> PreprocessedDigit:
    """Convert an uploaded digit into the 28x28, white-on-black format used by MNIST.

    Processing steps:
    1. Flatten transparency and convert to grayscale.
    2. Automatically invert common black-on-white uploads.
    3. Crop to the foreground bounding box.
    4. Preserve aspect ratio while fitting the digit inside a 20x20 box.
    5. Place it on a 28x28 canvas and center it by intensity centroid.
    6. Normalize to [0, 1] and flatten to shape (1, 784).
    """
    original = _open_image(source)
    flattened = _flatten_transparency(original)
    grayscale = ImageOps.grayscale(flattened)
    pixels = np.asarray(grayscale, dtype=np.uint8)

    was_inverted = bool(auto_invert and _border_mean(pixels) > 127)
    if was_inverted:
        pixels = 255 - pixels

    pixels = np.where(pixels > foreground_threshold, pixels, 0).astype(np.uint8)
    coords = np.argwhere(pixels > 0)
    if coords.size == 0:
        raise ValueError('No visible digit was detected. Use a clear image with strong contrast.')

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = Image.fromarray(pixels[y_min:y_max + 1, x_min:x_max + 1], mode='L')

    width, height = cropped.size
    scale = content_size / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cropped.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new('L', (28, 28), 0)
    paste_x = (28 - resized.width) // 2
    paste_y = (28 - resized.height) // 2
    canvas.paste(resized, (paste_x, paste_y))

    canvas_array = np.asarray(canvas, dtype=np.float32)
    mass = float(canvas_array.sum())
    if mass > 0:
        yy, xx = np.indices(canvas_array.shape)
        center_x = float((xx * canvas_array).sum() / mass)
        center_y = float((yy * canvas_array).sum() / mass)
        dx = int(round(13.5 - center_x))
        dy = int(round(13.5 - center_y))
        canvas = _translate(canvas, dx, dy)

    normalized = np.asarray(canvas, dtype=np.float32) / 255.0
    model_input = normalized.reshape(1, 784)

    return PreprocessedDigit(
        original=original.copy(),
        grayscale=grayscale,
        centered_28x28=canvas,
        model_input=model_input,
        was_inverted=was_inverted,
    )
