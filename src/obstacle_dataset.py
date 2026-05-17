from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image


DEFAULT_OBSTACLE_DATASET = "Abtinz/Obstacle-Detection-Dataset-YOLO"


@dataclass
class YoloBox:
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        x1 = (self.x_center - self.width / 2) * image_width
        y1 = (self.y_center - self.height / 2) * image_height
        x2 = (self.x_center + self.width / 2) * image_width
        y2 = (self.y_center + self.height / 2) * image_height
        return (
            int(max(0, min(image_width - 1, round(x1)))),
            int(max(0, min(image_height - 1, round(y1)))),
            int(max(0, min(image_width - 1, round(x2)))),
            int(max(0, min(image_height - 1, round(y2)))),
        )


def load_obstacle_rows(
    dataset_name: str,
    split: str,
    max_samples: int,
    min_objects: int = 1,
    random_seed: int | None = None,
):
    dataset = load_dataset(dataset_name, split=split)
    available_files = set(list_repo_files(dataset_name, repo_type="dataset"))
    selected = []
    for index, row in enumerate(dataset):
        if int(row.get("num_objects", 0)) < min_objects:
            continue
        if resolve_image_repo_path(row, split) in available_files:
            selected.append(index)
    if random_seed is not None:
        random.Random(random_seed).shuffle(selected)
    if not selected:
        raise ValueError(
            f"No downloadable image files found for split '{split}' with at least {min_objects} objects."
        )
    return dataset.select(selected[:max_samples])


def load_balanced_obstacle_rows(
    dataset_name: str,
    split: str,
    max_samples: int,
    single_fraction: float = 0.5,
    random_seed: int | None = None,
):
    dataset = load_dataset(dataset_name, split=split)
    available_files = set(list_repo_files(dataset_name, repo_type="dataset"))
    single_target = max(1, round(max_samples * single_fraction))
    multi_target = max(0, max_samples - single_target)
    singles = []
    multis = []

    for index, row in enumerate(dataset):
        if resolve_image_repo_path(row, split) not in available_files:
            continue
        num_objects = int(row.get("num_objects", 0))
        if num_objects == 1:
            singles.append(index)
        elif num_objects > 1:
            multis.append(index)

    rng = random.Random(random_seed)
    rng.shuffle(singles)
    rng.shuffle(multis)

    selected = singles[:single_target] + multis[:multi_target]
    if len(selected) < max_samples:
        selected_set = set(selected)
        fallback = singles[single_target:] + multis[multi_target:]
        rng.shuffle(fallback)
        for index in fallback:
            if index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            if len(selected) >= max_samples:
                break

    if not selected:
        raise ValueError(f"No downloadable image files found for split '{split}'.")
    return dataset.select(selected[:max_samples])


def parse_yolo_boxes(row: dict[str, Any]) -> list[YoloBox]:
    raw_boxes = json.loads(row["bboxes_yolo"]) if row.get("bboxes_yolo") else []
    class_names = [name.strip() for name in str(row.get("class_names", "")).split(",") if name.strip()]
    boxes = []
    for index, box in enumerate(raw_boxes):
        fallback_name = class_names[min(index, len(class_names) - 1)] if class_names else str(box["class_id"])
        boxes.append(
            YoloBox(
                class_id=int(box["class_id"]),
                class_name=box.get("class_name") or fallback_name,
                x_center=float(box["x_center"]),
                y_center=float(box["y_center"]),
                width=float(box["width"]),
                height=float(box["height"]),
            )
        )
    return boxes


def resolve_image_repo_path(row: dict[str, Any], split: str) -> str:
    file_name = str(row["file_name"])
    if file_name.startswith(f"{split}/"):
        return file_name
    if file_name.startswith("images/"):
        return f"{split}/{file_name}"
    return file_name


def load_obstacle_image(
    dataset_name: str,
    row: dict[str, Any],
    split: str,
    retries: int = 4,
    backoff_seconds: float = 1.5,
) -> Image.Image:
    repo_path = resolve_image_repo_path(row, split)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            local_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=repo_path)
            return Image.open(Path(local_path)).convert("RGB")
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    raise RuntimeError(f"Failed to download image '{repo_path}' after {retries} attempts.") from last_error
