from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from src.image_preprocessing import preprocess_digit_image


def test_black_digit_on_white_background_is_inverted_and_flattened() -> None:
    image = Image.new('L', (100, 100), 255)
    draw = ImageDraw.Draw(image)
    draw.line((50, 15, 50, 85), fill=0, width=12)

    result = preprocess_digit_image(image)

    assert result.was_inverted is True
    assert result.centered_28x28.size == (28, 28)
    assert result.model_input.shape == (1, 784)
    assert result.model_input.dtype == np.float32
    assert 0.0 <= float(result.model_input.min()) <= float(result.model_input.max()) <= 1.0
    assert float(result.model_input.sum()) > 0


def test_white_digit_on_black_background_is_not_inverted() -> None:
    image = Image.new('L', (100, 100), 0)
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 15, 75, 85), outline=255, width=10)

    result = preprocess_digit_image(image)

    assert result.was_inverted is False
    assert result.model_input.shape == (1, 784)


def test_blank_image_raises_clear_error() -> None:
    image = Image.new('L', (50, 50), 255)
    try:
        preprocess_digit_image(image)
    except ValueError as exc:
        assert 'No visible digit' in str(exc)
    else:
        raise AssertionError('Expected ValueError for a blank image.')
