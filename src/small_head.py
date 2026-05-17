from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .ijepa_localization import IJepaPatchLocalizer
from .obstacle_dataset import YoloBox, load_obstacle_image, parse_yolo_boxes
from .prototypes import safe_crop


@dataclass
class HeadGuess:
    object_index: int
    yolo_label: str
    head_guess: str
    confidence: float
    agreement: bool


@dataclass
class TrainedSmallHead:
    classifier: LogisticRegression
    classes: list[str]
    train_accuracy: float
    train_objects: int
    parameter_count: int


def train_small_head(
    dataset_name: str,
    split: str,
    rows,
    localizer: IJepaPatchLocalizer,
) -> TrainedSmallHead | None:
    embeddings = []
    labels = []
    for row in rows:
        image = load_obstacle_image(dataset_name, row, split)
        for box in parse_yolo_boxes(row):
            crop = safe_crop(image, box)
            if crop is None:
                continue
            embeddings.append(localizer.embed_image(crop))
            labels.append(box.class_name)

    if len(set(labels)) < 2 or len(labels) < 4:
        return None

    x = np.vstack(embeddings)
    y = np.asarray(labels)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(x, y)
    predictions = classifier.predict(x)
    return TrainedSmallHead(
        classifier=classifier,
        classes=list(classifier.classes_),
        train_accuracy=float(accuracy_score(y, predictions)),
        train_objects=int(len(labels)),
        parameter_count=int(classifier.coef_.size + classifier.intercept_.size),
    )


def guess_objects_with_head(
    image: Image.Image,
    yolo_boxes: list[YoloBox],
    localizer: IJepaPatchLocalizer,
    trained_head: TrainedSmallHead | None,
) -> list[HeadGuess]:
    if trained_head is None:
        return []

    guesses = []
    for index, box in enumerate(yolo_boxes, start=1):
        crop = safe_crop(image, box)
        if crop is None:
            continue
        embedding = localizer.embed_image(crop)
        probabilities = trained_head.classifier.predict_proba([embedding])[0]
        best_index = int(np.argmax(probabilities))
        guess = trained_head.classes[best_index]
        guesses.append(
            HeadGuess(
                object_index=index,
                yolo_label=box.class_name,
                head_guess=guess,
                confidence=float(probabilities[best_index]),
                agreement=guess == box.class_name,
            )
        )
    return guesses
