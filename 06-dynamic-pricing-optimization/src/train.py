"""Command-line entry point for reproducible training and evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .data_generation import generate_synthetic_pricing_data
from .model_training import train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the dynamic pricing ANN project.")
    parser.add_argument("--samples", type=int, default=16_000)
    parser.add_argument("--epochs-regression", type=int, default=20)
    parser.add_argument("--epochs-classification", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_pricing_data(args.samples)
    data.head(250).to_csv(data_dir / "sample_training_data.csv", index=False)
    train_models(
        data,
        model_dir,
        epochs_regression=args.epochs_regression,
        epochs_classification=args.epochs_classification,
        batch_size=args.batch_size,
    )

    env = os.environ.copy()
    env["KERAS_BACKEND"] = "torch"
    subprocess.run([sys.executable, "-m", "src.finalize"], cwd=project_root, env=env, check=True)
    print("Training and evaluation completed successfully.")


if __name__ == "__main__":
    main()
