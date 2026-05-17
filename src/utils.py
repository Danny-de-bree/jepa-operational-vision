from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset


def load_image_split(
    dataset_name: str,
    split: str,
    max_samples: int | None = None,
) -> Dataset:
    dataset = load_dataset(dataset_name, split=split)
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def detect_label_names(dataset: Dataset, label_column: str) -> list[str] | None:
    feature = dataset.features.get(label_column)
    if feature is not None and hasattr(feature, "names"):
        return list(feature.names)
    return None


def label_to_name(label: Any, label_names: list[str] | None) -> str:
    if label_names and isinstance(label, (int, np.integer)) and 0 <= int(label) < len(label_names):
        return label_names[int(label)]
    return str(label)


def embeddings_to_frame(
    embeddings: np.ndarray,
    labels: list[Any],
    label_names: list[str] | None,
    sample_ids: list[str],
    model_name: str,
    dataset_name: str,
    split: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "label_name": [label_to_name(label, label_names) for label in labels],
            "dataset_name": dataset_name,
            "split": split,
            "model_name": model_name,
            "embedding": [embedding.astype(float).tolist() for embedding in embeddings],
        }
    )


def save_embeddings(df: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / "embeddings.parquet"
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def read_embeddings(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"sample_id", "label", "embedding"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Embedding file is missing required columns: {sorted(missing)}")
    return df


def embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(df["embedding"].map(np.asarray).to_list()).astype(np.float32)
