from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from .ijepa_localization import IJepaPatchLocalizer
from .obstacle_dataset import YoloBox, load_obstacle_image, parse_yolo_boxes


@dataclass
class PrototypeGuess:
    object_index: int
    yolo_label: str
    ijepa_guess: str
    similarity: float
    agreement: bool


def build_class_prototypes(
    dataset_name: str,
    split: str,
    rows,
    localizer: IJepaPatchLocalizer,
) -> dict[str, np.ndarray]:
    embeddings_by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        image = load_obstacle_image(dataset_name, row, split)
        for box in parse_yolo_boxes(row):
            crop = safe_crop(image, box)
            if crop is None:
                continue
            embeddings_by_class[box.class_name].append(localizer.embed_image(crop))

    return {
        class_name: np.vstack(embeddings).mean(axis=0)
        for class_name, embeddings in embeddings_by_class.items()
        if embeddings
    }


def guess_objects_with_prototypes(
    image: Image.Image,
    yolo_boxes: list[YoloBox],
    localizer: IJepaPatchLocalizer,
    prototypes: dict[str, np.ndarray],
) -> list[PrototypeGuess]:
    if not prototypes:
        return []

    class_names = list(prototypes)
    prototype_matrix = np.vstack([prototypes[class_name] for class_name in class_names])
    guesses = []
    for index, box in enumerate(yolo_boxes, start=1):
        crop = safe_crop(image, box)
        if crop is None:
            continue
        embedding = localizer.embed_image(crop)
        similarities = cosine_similarity([embedding], prototype_matrix)[0]
        best_index = int(np.argmax(similarities))
        guess = class_names[best_index]
        guesses.append(
            PrototypeGuess(
                object_index=index,
                yolo_label=box.class_name,
                ijepa_guess=guess,
                similarity=float(similarities[best_index]),
                agreement=guess == box.class_name,
            )
        )
    return guesses


def safe_crop(image: Image.Image, box: YoloBox) -> Image.Image | None:
    x1, y1, x2, y2 = box.to_xyxy(*image.size)
    image_width, image_height = image.size
    x1 = max(0, min(image_width - 1, x1))
    x2 = max(0, min(image_width, x2))
    y1 = max(0, min(image_height - 1, y1))
    y2 = max(0, min(image_height, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image.crop((x1, y1, x2, y2))
