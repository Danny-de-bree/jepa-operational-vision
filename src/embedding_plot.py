from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .degradation import DegradationConfig, degrade_image
from .ijepa_localization import IJepaPatchLocalizer
from .obstacle_dataset import DEFAULT_OBSTACLE_DATASET, load_balanced_obstacle_rows, load_obstacle_image, parse_yolo_boxes
from .prototypes import safe_crop


def build_embedding_plot_data(
    dataset_name: str,
    split: str,
    model_name: str,
    samples: int,
    seed: int,
    method: str,
    output: str,
    from_bulk: str | None = None,
    compare_bulk: str | None = None,
) -> Path:
    if compare_bulk:
        if not from_bulk:
            raise ValueError("--compare-bulk requires --from-bulk as the baseline run.")
        return build_compare_projection(
            dataset_name=dataset_name,
            split=split,
            model_name=model_name,
            baseline_bulk=from_bulk,
            compare_bulk=compare_bulk,
            method=method,
            seed=seed,
            output=output,
        )

    bulk_df = pd.read_csv(from_bulk) if from_bulk else None
    if bulk_df is not None and is_crop_robustness_csv(bulk_df):
        return build_crop_projection_from_bulk(
            dataset_name=dataset_name,
            split=split,
            model_name=model_name,
            bulk_df=bulk_df,
            method=method,
            seed=seed,
            output=output,
        )

    rows = load_rows_from_bulk(dataset_name, split, bulk_df) if bulk_df is not None else load_balanced_obstacle_rows(
        dataset_name, split, samples, random_seed=seed
    )
    localizer = IJepaPatchLocalizer(model_name=model_name)
    records = []

    for sample_index, row in enumerate(rows):
        image = load_obstacle_image(dataset_name, row, split)
        boxes = parse_yolo_boxes(row)
        allowed_objects = None
        bulk_by_object = {}
        if bulk_df is not None:
            row_bulk = bulk_df[bulk_df["file_name"] == row["file_name"]]
            allowed_objects = set(row_bulk["object"].astype(int))
            bulk_by_object = {int(item["object"]): item for item in row_bulk.to_dict(orient="records")}
            image = apply_bulk_degradation(image, row_bulk)
        for object_index, box in enumerate(boxes, start=1):
            if allowed_objects is not None and object_index not in allowed_objects:
                continue
            crop = safe_crop(image, box)
            if crop is None:
                continue
            embedding = localizer.embed_image(crop)
            record = {
                "sample": sample_index,
                "file_name": row["file_name"],
                "object": object_index,
                "label": box.class_name,
                "embedding": embedding,
            }
            if object_index in bulk_by_object:
                record.update(
                    {
                        key: value
                        for key, value in bulk_by_object[object_index].items()
                        if key not in {"sample", "file_name", "object"}
                    }
                )
            records.append(record)

    if len(records) < 3:
        raise ValueError("Need at least 3 object embeddings for a 2D projection.")

    embeddings = [record.pop("embedding") for record in records]
    projected = project_embeddings(embeddings, method=method, seed=seed)
    for record, point in zip(records, projected):
        record["x"] = float(point[0])
        record["y"] = float(point[1])

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return output_path


def build_crop_projection_from_bulk(
    dataset_name: str,
    split: str,
    model_name: str,
    bulk_df: pd.DataFrame,
    method: str,
    seed: int,
    output: str,
) -> Path:
    localizer = IJepaPatchLocalizer(model_name=model_name)
    rows = load_rows_from_bulk(dataset_name, split, bulk_df)
    records = []

    for sample_index, row in enumerate(rows):
        image = load_obstacle_image(dataset_name, row, split)
        row_bulk = bulk_df[bulk_df["file_name"] == row["file_name"]]
        boxes = parse_yolo_boxes(row)
        for bulk_record in row_bulk.to_dict(orient="records"):
            object_index = int(bulk_record["object"])
            if object_index < 1 or object_index > len(boxes):
                continue
            crop = safe_crop(image, boxes[object_index - 1])
            if crop is None:
                continue
            crop = apply_crop_degradation(crop, bulk_record)
            embedding = localizer.embed_image(crop)
            record = {
                "sample": sample_index,
                "file_name": row["file_name"],
                "object": object_index,
                "label": boxes[object_index - 1].class_name,
                "embedding": embedding,
            }
            record.update(
                {
                    key: value
                    for key, value in bulk_record.items()
                    if key not in {"sample", "file_name", "object"}
                }
            )
            records.append(record)

    if len(records) < 3:
        raise ValueError("Need at least 3 object embeddings for a 2D projection.")

    embeddings = [record.pop("embedding") for record in records]
    projected = project_embeddings(embeddings, method=method, seed=seed)
    for record, point in zip(records, projected):
        record["x"] = float(point[0])
        record["y"] = float(point[1])

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return output_path


def build_compare_projection(
    dataset_name: str,
    split: str,
    model_name: str,
    baseline_bulk: str,
    compare_bulk: str,
    method: str,
    seed: int,
    output: str,
) -> Path:
    baseline_df = pd.read_csv(baseline_bulk)
    compare_df = pd.read_csv(compare_bulk)
    localizer = IJepaPatchLocalizer(model_name=model_name)
    records = []

    for condition, bulk_df in [("baseline", baseline_df), ("compare", compare_df)]:
        rows = load_rows_from_bulk(dataset_name, split, bulk_df)
        for sample_index, row in enumerate(rows):
            image = load_obstacle_image(dataset_name, row, split)
            row_bulk = bulk_df[bulk_df["file_name"] == row["file_name"]]
            image = apply_bulk_degradation(image, row_bulk)
            allowed_objects = set(row_bulk["object"].astype(int))
            bulk_by_object = {int(item["object"]): item for item in row_bulk.to_dict(orient="records")}
            for object_index, box in enumerate(parse_yolo_boxes(row), start=1):
                if object_index not in allowed_objects:
                    continue
                crop = safe_crop(image, box)
                if crop is None:
                    continue
                embedding = localizer.embed_image(crop)
                bulk_record = bulk_by_object.get(object_index, {})
                record = {
                    "condition": condition,
                    "sample": sample_index,
                    "file_name": row["file_name"],
                    "object": object_index,
                    "label": box.class_name,
                    "embedding": embedding,
                }
                record.update(
                    {
                        key: value
                        for key, value in bulk_record.items()
                        if key not in {"sample", "file_name", "object"}
                    }
                )
                records.append(record)

    if len(records) < 3:
        raise ValueError("Need at least 3 object embeddings for a 2D projection.")

    embeddings = [record.pop("embedding") for record in records]
    projected = project_embeddings(embeddings, method=method, seed=seed)
    for record, point in zip(records, projected):
        record["x"] = float(point[0])
        record["y"] = float(point[1])

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return output_path


def load_rows_from_bulk(dataset_name: str, split: str, bulk_df: pd.DataFrame):
    wanted_files = set(bulk_df["file_name"].astype(str))
    rows = []
    dataset = load_dataset(dataset_name, split=split)
    for row in dataset:
        if row["file_name"] in wanted_files:
            rows.append(row)
        if len({row["file_name"] for row in rows}) >= len(wanted_files):
            break
    missing = wanted_files.difference({row["file_name"] for row in rows})
    if missing:
        raise ValueError(f"Could not resolve {len(missing)} files from bulk CSV, including: {sorted(missing)[:5]}")
    return rows


def apply_bulk_degradation(image, row_bulk: pd.DataFrame):
    if row_bulk.empty or "degradation_mode" not in row_bulk:
        return image
    first = row_bulk.iloc[0]
    mode = str(first.get("degradation_mode", "none"))
    if mode == "none":
        return image
    config = DegradationConfig(
        mode=mode,
        ratio=float(first.get("degradation_ratio", 0.0)),
        patch_size=int(first.get("degradation_patch_size", 32)),
        fill=str(first.get("degradation_fill", "gray")),
        seed=int(first.get("seed", 7)),
    )
    sample_index = int(first.get("sample", 0))
    return degrade_image(image, config, sample_index=sample_index)


def apply_crop_degradation(crop, bulk_record: dict):
    mode = str(bulk_record.get("degradation_mode", "none"))
    ratio = float(bulk_record.get("degradation_ratio", 0.0))
    if mode == "none" or ratio <= 0:
        return crop.convert("RGB")
    config = DegradationConfig(
        mode=mode,
        ratio=ratio,
        patch_size=int(bulk_record.get("degradation_patch_size", 32)),
        fill=str(bulk_record.get("degradation_fill", "gray")),
        seed=int(bulk_record.get("seed", 7)),
    )
    sample_index = int(bulk_record.get("sample", 0)) * 10_000 + int(bulk_record.get("object", 0)) * 100 + int(
        ratio * 100
    )
    return degrade_image(crop, config, sample_index=sample_index)


def is_crop_robustness_csv(df: pd.DataFrame) -> bool:
    if "degradation_scope" not in df.columns:
        return False
    return bool((df["degradation_scope"].astype(str) == "crop").any())


def project_embeddings(embeddings, method: str, seed: int):
    matrix = np.vstack(embeddings)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(matrix)
    perplexity = max(2, min(30, (matrix.shape[0] - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(matrix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export 2D I-JEPA object embedding projection data.")
    parser.add_argument("--dataset-name", default=DEFAULT_OBSTACLE_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-name", default="facebook/ijepa_vith14_1k")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--output", default="outputs/embedding_projection.csv")
    parser.add_argument("--from-bulk", default=None, help="Use exact objects from a bulk_eval objects.csv file.")
    parser.add_argument(
        "--compare-bulk",
        default=None,
        help="Project --from-bulk and this second bulk objects.csv into one shared 2D space.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = build_embedding_plot_data(**vars(args))
    print(f"Saved embedding projection data: {output}")


if __name__ == "__main__":
    main()
