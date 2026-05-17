from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .obstacle_dataset import YoloBox


def draw_yolo_with_heatmap(
    image: Image.Image,
    yolo_boxes: list[YoloBox],
    heatmap: np.ndarray,
    saliency_threshold: float = 0.7,
) -> Image.Image:
    canvas = overlay_heatmap(image, heatmap, saliency_threshold=saliency_threshold)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for box in yolo_boxes:
        xyxy = box.to_xyxy(*canvas.size)
        draw.rectangle(xyxy, outline="lime", width=4)
        draw_label(draw, xyxy, f"YOLO: {box.class_name}", "lime", font)

    return canvas


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    saliency_threshold: float = 0.7,
) -> Image.Image:
    base = image.convert("RGB")
    heat = np.clip(heatmap, 0, 1)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8), mode="L").resize(base.size)
    heat_values = np.asarray(heat_img).astype(np.float32) / 255.0
    alpha_values = np.where(
        heat_values >= saliency_threshold,
        ((heat_values - saliency_threshold) / max(1e-6, 1 - saliency_threshold)) * 210,
        0,
    )
    alpha = Image.fromarray(alpha_values.astype(np.uint8), mode="L")

    color = Image.new("RGBA", base.size, (255, 20, 0, 0))
    color.putalpha(alpha)
    return Image.alpha_composite(base.convert("RGBA"), color).convert("RGB")


def draw_detection_overlay(
    image: Image.Image,
    yolo_boxes: list[YoloBox],
    ijepa_box: tuple[int, int, int, int] | None = None,
    ijepa_candidate_boxes: list[tuple[int, int, int, int]] | None = None,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for box in yolo_boxes:
        xyxy = box.to_xyxy(*canvas.size)
        draw.rectangle(xyxy, outline="lime", width=4)
        draw_label(draw, xyxy, f"YOLO: {box.class_name}", "lime", font)

    for index, candidate_box in enumerate(ijepa_candidate_boxes or [], start=1):
        draw.rectangle(candidate_box, outline="red", width=3)
        draw_label(draw, candidate_box, f"I-JEPA {index}", "red", font)

    if ijepa_box is not None:
        draw.rectangle(ijepa_box, outline="red", width=4)
        draw_label(draw, ijepa_box, "I-JEPA estimate", "red", font)

    return canvas


def draw_label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, color: str, font) -> None:
    x1, y1, _, _ = box
    text_box = draw.textbbox((x1, y1), text, font=font)
    text_h = text_box[3] - text_box[1]
    label_box = (x1, max(0, y1 - text_h - 6), x1 + text_box[2] - text_box[0] + 8, max(text_h + 6, y1))
    draw.rectangle(label_box, fill=color)
    draw.text((label_box[0] + 4, label_box[1] + 3), text, fill="black", font=font)
