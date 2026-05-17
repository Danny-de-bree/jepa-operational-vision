from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from .ijepa_localization import IJepaPatchLocalizer
from .obstacle_dataset import YoloBox


@dataclass
class ObjectContextResult:
    object_index: int
    class_name: str
    object_context_similarity: float
    scene_context_similarity: float
    context_pattern: str
    context_strength: str
    object_box_xyxy: tuple[int, int, int, int]
    context_box_xyxy: tuple[int, int, int, int]


def analyze_object_contexts(
    image: Image.Image,
    yolo_boxes: list[YoloBox],
    localizer: IJepaPatchLocalizer,
    margin_ratio: float = 0.75,
) -> list[ObjectContextResult]:
    if not yolo_boxes:
        return []

    scene_embedding = localizer.embed_image(image)
    results = []
    for index, box in enumerate(yolo_boxes, start=1):
        object_box = box.to_xyxy(*image.size)
        context_box = expand_box(object_box, image.size, margin_ratio=margin_ratio)
        object_crop = image.crop(object_box)
        context_crop = image.crop(context_box)
        if object_crop.width < 2 or object_crop.height < 2 or context_crop.width < 2 or context_crop.height < 2:
            continue
        object_embedding = localizer.embed_image(object_crop)
        context_embedding = localizer.embed_image(context_crop)

        object_context_similarity = vector_similarity(object_embedding, context_embedding)
        scene_context_similarity = vector_similarity(scene_embedding, context_embedding)
        results.append(
            ObjectContextResult(
                object_index=index,
                class_name=box.class_name,
                object_context_similarity=object_context_similarity,
                scene_context_similarity=scene_context_similarity,
                context_pattern=describe_context_pattern(
                    object_context_similarity,
                    scene_context_similarity,
                ),
                context_strength=describe_context_strength(
                    object_context_similarity,
                    scene_context_similarity,
                ),
                object_box_xyxy=object_box,
                context_box_xyxy=context_box,
            )
        )
    return results


def expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    return (
        int(max(0, x1 - margin_x)),
        int(max(0, y1 - margin_y)),
        int(min(image_width - 1, x2 + margin_x)),
        int(min(image_height - 1, y2 + margin_y)),
    )


def vector_similarity(left, right) -> float:
    return float(cosine_similarity([left], [right])[0, 0])


def describe_context_pattern(object_context_similarity: float, scene_context_similarity: float) -> str:
    if object_context_similarity >= 0.92 and scene_context_similarity >= 0.92:
        return "near other objects / scene-embedded"
    if object_context_similarity >= 0.9:
        return "isolated / object-dominant"
    if scene_context_similarity >= 0.9:
        return "group / crowd context"
    return "distinct object/context"


def describe_context_strength(object_context_similarity: float, scene_context_similarity: float) -> str:
    strength = max(object_context_similarity, scene_context_similarity)
    if strength >= 0.93:
        return "high"
    if strength >= 0.86:
        return "medium"
    return "low"
