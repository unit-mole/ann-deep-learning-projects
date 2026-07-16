from __future__ import annotations

import json
import pickle
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    MODEL_DIR,
    NUMERICAL_FEATURES,
    OUTPUT_DIR,
    SEED,
    TARGET_COLUMN,
)
from src.data_generation import generate_synthetic_tabular_data
from src.data_preprocessing import build_model_inputs, fit_all_preprocessors, split_dataset
from src.embedding_analysis import extract_embedding_table, plot_embedding_pca
from src.embedding_preprocessing import choose_embedding_dimension
from src.model_evaluation import (
    evaluate_binary_classifier,
    save_evaluation_plots,
    save_metrics,
    save_model_comparison,
    save_permutation_importance_plot,
    save_prediction_distribution,
    save_training_curves,
    select_f1_threshold,
)
from src.model_training import (
    build_tabular_embedding_model,
    compile_binary_model,
    compute_class_weights,
    train_embedding_model,
    train_random_forest_baseline,
)

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def _build_numeric_metadata(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, list[float]]]:
    defaults: dict[str, float] = {}
    ranges: dict[str, list[float]] = {}
    for feature in NUMERICAL_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        defaults[feature] = float(values.median())
        ranges[feature] = [float(values.min()), float(values.max())]
    defaults["age"] = int(round(defaults["age"]))
    ranges["age"] = [int(round(ranges["age"][0])), int(round(ranges["age"][1]))]
    return defaults, ranges


def _raw_feature_permutation_importance(
    model,
    validation_frame: pd.DataFrame,
    validation_target: np.ndarray,
    encoders: dict[str, object],
    imputer,
    scaler,
) -> pd.DataFrame:
    """Measure validation ROC-AUC decrease after shuffling one raw column."""
    base_inputs, _ = build_model_inputs(
        validation_frame,
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
        encoders,
        imputer,
        scaler,
    )
    base_probability = model.predict(base_inputs, verbose=0).reshape(-1)
    base_auc = roc_auc_score(validation_target, base_probability)
    rng = np.random.default_rng(SEED)
    rows = []

    for feature in FEATURE_COLUMNS:
        permuted = validation_frame[FEATURE_COLUMNS].copy()
        permuted[feature] = rng.permutation(permuted[feature].to_numpy())
        permuted_inputs, _ = build_model_inputs(
            permuted,
            CATEGORICAL_FEATURES,
            NUMERICAL_FEATURES,
            encoders,
            imputer,
            scaler,
        )
        probability = model.predict(permuted_inputs, verbose=0).reshape(-1)
        rows.append(
            {
                "feature": feature,
                "roc_auc_decrease": float(base_auc - roc_auc_score(validation_target, probability)),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc_decrease", ascending=False)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_tabular_data(seed=SEED)
    splits = split_dataset(data, TARGET_COLUMN, seed=SEED)

    # All learned preprocessing objects are fitted on the training partition only.
    encoders, imputer, scaler = fit_all_preprocessors(
        splits.train, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
    )
    train_inputs, _ = build_model_inputs(
        splits.train,
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
        encoders,
        imputer,
        scaler,
    )
    validation_inputs, _ = build_model_inputs(
        splits.validation,
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
        encoders,
        imputer,
        scaler,
    )
    test_inputs, _ = build_model_inputs(
        splits.test,
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
        encoders,
        imputer,
        scaler,
    )

    model = build_tabular_embedding_model(
        CATEGORICAL_FEATURES,
        encoders,
        len(NUMERICAL_FEATURES),
    )
    compile_binary_model(model)
    history = train_embedding_model(
        model,
        train_inputs,
        splits.train[TARGET_COLUMN].to_numpy(dtype="float32"),
        validation_inputs,
        splits.validation[TARGET_COLUMN].to_numpy(dtype="float32"),
        compute_class_weights(splits.train[TARGET_COLUMN]),
    )

    validation_target = splits.validation[TARGET_COLUMN].to_numpy()
    test_target = splits.test[TARGET_COLUMN].to_numpy()
    validation_probability = model.predict(validation_inputs, verbose=0).reshape(-1)
    test_probability = model.predict(test_inputs, verbose=0).reshape(-1)

    best_threshold, threshold_table = select_f1_threshold(
        validation_target,
        validation_probability,
    )
    ann_validation_default = evaluate_binary_classifier(
        validation_target,
        validation_probability,
        threshold=0.50,
    )
    ann_validation_best_f1 = evaluate_binary_classifier(
        validation_target,
        validation_probability,
        threshold=best_threshold,
    )
    ann_test_default = evaluate_binary_classifier(
        test_target,
        test_probability,
        threshold=0.50,
    )

    baseline = train_random_forest_baseline(
        splits.train[FEATURE_COLUMNS],
        splits.train[TARGET_COLUMN],
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
        seed=SEED,
    )
    baseline_probability = baseline.predict_proba(splits.test[FEATURE_COLUMNS])[:, 1]
    baseline_metrics = evaluate_binary_classifier(
        test_target,
        baseline_probability,
        threshold=0.50,
    )

    model.save(MODEL_DIR / "tabular_embedding_ann.keras")
    with open(MODEL_DIR / "label_encoders.pkl", "wb") as file:
        pickle.dump(encoders, file)
    with open(MODEL_DIR / "numeric_imputer.pkl", "wb") as file:
        pickle.dump(imputer, file)
    with open(MODEL_DIR / "numeric_scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    numeric_defaults, numeric_ranges = _build_numeric_metadata(splits.train)
    embedding_shapes = {
        feature: list(model.get_layer(f"{feature}_embedding").get_weights()[0].shape)
        for feature in CATEGORICAL_FEATURES
    }
    metadata = {
        "project_name": "Tabular Deep Learning with Embeddings",
        "task_type": "binary_classification",
        "target_column": TARGET_COLUMN,
        "target_description": "Synthetic positive business-outcome propensity",
        "class_labels": {"0": "Lower propensity", "1": "Higher propensity"},
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "default_threshold": 0.50,
        "validation_f1_threshold": best_threshold,
        "embedding_dimension_rule": "min(50, ceil(sqrt(vocabulary_size)) + 1)",
        "encoder_strategy": "training_only_safe_encoder_with_oov_index_0",
        "unknown_category_policy": "Map unseen categories to reserved OOV index 0.",
        "fallback_categories": {},
        "numeric_defaults": numeric_defaults,
        "numeric_ranges": numeric_ranges,
        "model_file": "tabular_embedding_ann.keras",
        "model_parameter_count": int(model.count_params()),
        "embedding_dimensions": {
            feature: choose_embedding_dimension(encoders[feature].vocab_size)
            for feature in CATEGORICAL_FEATURES
        },
        "embedding_shapes": embedding_shapes,
        "training_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "test_rows": len(splits.test),
        "dataset_rows": len(data),
        "dataset_source": "Deterministic synthetic generator included in src/data_generation.py",
        "artifact_compatibility": {
            "keras_version_saved": version("keras"),
            "scikit_learn_version_saved": version("scikit-learn"),
        },
    }
    (MODEL_DIR / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    comparison = pd.DataFrame(
        [
            {"model": "Embedding ANN", **{key: ann_test_default[key] for key in [
                "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss"
            ]}},
            {"model": "Random Forest", **{key: baseline_metrics[key] for key in [
                "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss"
            ]}},
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "final_metrics_summary.csv", index=False)

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    threshold_table.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)

    metrics_payload = {
        "dataset": {
            "rows": len(data),
            "positive_rate": float(data[TARGET_COLUMN].mean()),
            "train_rows": len(splits.train),
            "validation_rows": len(splits.validation),
            "test_rows": len(splits.test),
        },
        "embedding_ann": {
            "validation_default": ann_validation_default,
            "validation_best_f1": ann_validation_best_f1,
            "test_default": ann_test_default,
        },
        "random_forest_baseline": baseline_metrics,
        "model_parameter_count": int(model.count_params()),
        "embedding_shapes": embedding_shapes,
        "methodology_note": (
            "All preprocessors fit on training data; threshold selected on validation data; "
            "final test metrics reported at threshold 0.50."
        ),
    }
    save_metrics(metrics_payload, OUTPUT_DIR / "model_metrics.json")

    save_evaluation_plots(test_target, test_probability, OUTPUT_DIR, threshold=0.50)
    save_training_curves(history.history, OUTPUT_DIR / "training_curves.png")
    save_model_comparison(comparison, OUTPUT_DIR / "model_comparison.png")
    save_prediction_distribution(
        test_probability,
        OUTPUT_DIR / "sample_prediction_distribution.png",
    )

    importance = _raw_feature_permutation_importance(
        model,
        splits.validation,
        validation_target,
        encoders,
        imputer,
        scaler,
    )
    importance.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)
    save_permutation_importance_plot(
        importance,
        OUTPUT_DIR / "permutation_importance.png",
    )

    embedding_table = extract_embedding_table(model, "occupation", encoders["occupation"])
    embedding_table.to_csv(OUTPUT_DIR / "occupation_embedding_vectors.csv")
    plot_embedding_pca(
        embedding_table,
        "Occupation Embeddings — PCA Projection",
        OUTPUT_DIR / "embedding_visualization.png",
    )

    sample = splits.test[FEATURE_COLUMNS].head(10).reset_index(drop=True)
    sample.to_csv(DATA_DIR / "sample_input.csv", index=False)
    sample_predictions = sample.copy()
    sample_predictions["actual_target"] = splits.test[TARGET_COLUMN].head(10).to_numpy()
    sample_predictions["prediction_probability"] = test_probability[:10]
    sample_predictions["predicted_class"] = (test_probability[:10] >= 0.50).astype(int)
    sample_predictions.to_csv(OUTPUT_DIR / "sample_predictions.csv", index=False)

    print("Training complete.")
    print(f"Saved model to: {MODEL_DIR / 'tabular_embedding_ann.keras'}")
    print(f"Validation-selected F1 threshold: {best_threshold:.2f}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
