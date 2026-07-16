from __future__ import annotations

import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_artifacts_exist() -> None:
    required = [
        PROJECT_ROOT / 'models/digit_recognition_model.keras',
        PROJECT_ROOT / 'models/best_params.json',
        PROJECT_ROOT / 'outputs/model_metrics.json',
        PROJECT_ROOT / 'app/streamlit_app.py',
    ]
    assert all(path.exists() for path in required)


def test_saved_model_is_valid_keras_archive() -> None:
    model_path = PROJECT_ROOT / 'models/digit_recognition_model.keras'
    with zipfile.ZipFile(model_path) as archive:
        names = set(archive.namelist())
    assert {'metadata.json', 'config.json', 'model.weights.h5'} <= names


def test_metrics_are_consistent() -> None:
    metrics = json.loads((PROJECT_ROOT / 'outputs/model_metrics.json').read_text())
    assert metrics['test_accuracy'] == 0.9824
    assert metrics['correct_predictions'] + metrics['misclassified_predictions'] == 10000
    assert metrics['best_hyperparameters']['hidden_units'] == 384
