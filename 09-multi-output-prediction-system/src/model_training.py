from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import RAW_FEATURES, SEED
from .data_generation import generate_synthetic_customer_data
from .data_preprocessing import build_preprocessor, to_dense
from .feature_engineering import add_deployment_features, fit_feature_metadata
from .model_evaluation import evaluate_outputs, save_metrics, select_threshold
from .target_preprocessing import fit_regression_scalers, inverse_scale, scale_regression_targets


def build_model(input_dim: int):
    import tensorflow as tf
    from tensorflow.keras import Model, layers
    inputs = layers.Input(shape=(input_dim,), name="customer_features")
    x = layers.Dense(256, activation="relu", name="shared_dense_1")(inputs)
    x = layers.BatchNormalization(name="shared_bn_1")(x)
    x = layers.Dropout(.30, name="shared_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="shared_dense_2")(x)
    x = layers.BatchNormalization(name="shared_bn_2")(x)
    x = layers.Dropout(.25, name="shared_dropout_2")(x)
    shared = layers.Dense(64, activation="relu", name="shared_repr")(x)
    churn = layers.Dense(32, activation="relu", name="churn_dense_1")(shared)
    churn = layers.Dropout(.20, name="churn_dropout_1")(churn)
    churn = layers.Dense(1, activation="sigmoid", name="churn_output")(churn)
    clv = layers.Dense(32, activation="relu", name="clv_dense_1")(shared)
    clv = layers.Dropout(.15, name="clv_dropout_1")(clv)
    clv = layers.Dense(1, activation="linear", name="clv_output")(clv)
    engagement = layers.Dense(32, activation="relu", name="eng_dense_1")(shared)
    engagement = layers.Dropout(.15, name="eng_dropout_1")(engagement)
    engagement = layers.Dense(1, activation="linear", name="engagement_output")(engagement)
    model = Model(inputs, [churn, clv, engagement], name="multi_output_prediction_ann")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
        loss={"churn_output":"binary_crossentropy","clv_output":"huber","engagement_output":"huber"},
        loss_weights={"churn_output":1.2,"clv_output":.8,"engagement_output":.8},
        metrics={
            "churn_output":[tf.keras.metrics.BinaryAccuracy(name="accuracy"),tf.keras.metrics.AUC(name="auc"),tf.keras.metrics.AUC(name="pr_auc",curve="PR")],
            "clv_output":[tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "engagement_output":[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    return model


def train(project_root: Path, epochs: int = 40) -> dict:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED); np.random.seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)
    df = generate_synthetic_customer_data()
    train_full, test = train_test_split(df, test_size=.20, random_state=SEED, stratify=df["churn"])
    train, validation = train_test_split(train_full, test_size=.20, random_state=SEED, stratify=train_full["churn"])
    feature_metadata = fit_feature_metadata(train)
    X_train = add_deployment_features(train[RAW_FEATURES], feature_metadata)
    X_validation = add_deployment_features(validation[RAW_FEATURES], feature_metadata)
    X_test = add_deployment_features(test[RAW_FEATURES], feature_metadata)
    preprocessor = build_preprocessor()
    X_train_nn = to_dense(preprocessor.fit_transform(X_train)).astype("float32")
    X_validation_nn = to_dense(preprocessor.transform(X_validation)).astype("float32")
    X_test_nn = to_dense(preprocessor.transform(X_test)).astype("float32")
    target_metadata = fit_regression_scalers(train["clv"].to_numpy(), train["engagement_score"].to_numpy())
    target_metadata.update(feature_metadata.to_dict())
    y_clv_train = scale_regression_targets(train["clv"].to_numpy(), target_metadata["clv_mean"], target_metadata["clv_std"])
    y_clv_validation = scale_regression_targets(validation["clv"].to_numpy(), target_metadata["clv_mean"], target_metadata["clv_std"])
    y_eng_train = scale_regression_targets(train["engagement_score"].to_numpy(), target_metadata["engagement_mean"], target_metadata["engagement_std"])
    y_eng_validation = scale_regression_targets(validation["engagement_score"].to_numpy(), target_metadata["engagement_mean"], target_metadata["engagement_std"])
    class_counts = train["churn"].value_counts().sort_index()
    class_weight = {0: len(train)/(2*class_counts[0]), 1: len(train)/(2*class_counts[1])}
    sample_weights = [
        np.where(train["churn"].to_numpy()==1,class_weight[1],class_weight[0]).astype("float32"),
        np.ones(len(train),dtype="float32"),np.ones(len(train),dtype="float32"),
    ]
    model = build_model(X_train_nn.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=4,min_lr=1e-5),
    ]
    model.fit(
        X_train_nn,[train["churn"].to_numpy(),y_clv_train,y_eng_train],
        validation_data=(X_validation_nn,[validation["churn"].to_numpy(),y_clv_validation,y_eng_validation]),
        sample_weight=sample_weights,epochs=epochs,batch_size=128,callbacks=callbacks,verbose=1,
    )
    val_churn, _, _ = model.predict(X_validation_nn, verbose=0)
    threshold = select_threshold(validation["churn"].to_numpy(), val_churn.reshape(-1))
    pred_churn, pred_clv_scaled, pred_eng_scaled = model.predict(X_test_nn, verbose=0)
    pred_clv = inverse_scale(pred_clv_scaled.reshape(-1), target_metadata["clv_mean"], target_metadata["clv_std"])
    pred_eng = inverse_scale(pred_eng_scaled.reshape(-1), target_metadata["engagement_mean"], target_metadata["engagement_std"])
    metrics = evaluate_outputs(test["churn"].to_numpy(), pred_churn.reshape(-1), test["clv"].to_numpy(), pred_clv, test["engagement_score"].to_numpy(), pred_eng, threshold)
    models_dir, outputs_dir = project_root/"models", project_root/"outputs"
    models_dir.mkdir(exist_ok=True); outputs_dir.mkdir(exist_ok=True)
    model.save(models_dir/"multi_output_model.keras")
    joblib.dump(preprocessor, models_dir/"preprocessor.joblib")
    target_metadata["churn_threshold"] = threshold
    (models_dir/"target_metadata.json").write_text(json.dumps(target_metadata,indent=2),encoding="utf-8")
    save_metrics(metrics, outputs_dir/"model_metrics.json")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(train(root, epochs=args.epochs), indent=2))
