from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def extract_embedding_table(model, feature: str, encoder: object) -> pd.DataFrame:
    weights = model.get_layer(f"{feature}_embedding").get_weights()[0]
    if hasattr(encoder, "categories_"):
        labels = ["__UNKNOWN__"] + list(encoder.categories_)
    elif hasattr(encoder, "classes_"):
        labels = [str(value) for value in encoder.classes_]
    else:
        labels = [f"category_{index}" for index in range(weights.shape[0])]
    return pd.DataFrame(weights, index=labels)


def nearest_categories(embedding_table: pd.DataFrame, category: str, top_k: int = 3) -> pd.DataFrame:
    if category not in embedding_table.index:
        raise KeyError(f"Unknown category: {category}")
    scores = cosine_similarity(
        embedding_table.loc[[category]].to_numpy(),
        embedding_table.to_numpy(),
    )[0]
    result = pd.DataFrame({"category": embedding_table.index, "cosine_similarity": scores})
    return result[result["category"] != category].nlargest(top_k, "cosine_similarity")


def plot_embedding_pca(
    embedding_table: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    if len(embedding_table) < 3:
        return
    coordinates = PCA(n_components=2, random_state=42).fit_transform(embedding_table.to_numpy())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=80)
    for label, (x_value, y_value) in zip(embedding_table.index, coordinates):
        ax.annotate(str(label), (x_value, y_value), xytext=(5, 5), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
