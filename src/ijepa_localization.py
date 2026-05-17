from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel


@dataclass
class LocalizationResult:
    box_xyxy: tuple[int, int, int, int]
    candidate_boxes_xyxy: list[tuple[int, int, int, int]]
    heatmap: np.ndarray
    score: float
    image_embedding: np.ndarray


class IJepaPatchLocalizer:
    """Patch-similarity localizer for I-JEPA-style encoders.

    I-JEPA is not an object detector. This class uses its patch embeddings as a
    representation probe: patches most similar to the image-level embedding are
    treated as the likely salient object region.
    """

    def __init__(self, model_name: str = "facebook/ijepa_vith14_1k", device: str | None = None):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def localize(
        self,
        image: Image.Image,
        threshold_quantile: float = 0.85,
        max_boxes: int = 8,
    ) -> LocalizationResult:
        width, height = image.size
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden = outputs.last_hidden_state[0].detach().cpu().float().numpy()
        patch_embeddings = self._patch_tokens(hidden)
        grid_size = int(np.sqrt(len(patch_embeddings)))
        if grid_size * grid_size != len(patch_embeddings):
            raise ValueError(f"Cannot reshape {len(patch_embeddings)} patch tokens into a square grid.")

        image_embedding = patch_embeddings.mean(axis=0, keepdims=True)
        scores = cosine_similarity(patch_embeddings, image_embedding).reshape(grid_size, grid_size)
        heatmap = normalize(scores)
        box = heatmap_to_box(heatmap, width, height, threshold_quantile)
        candidate_boxes = heatmap_to_connected_boxes(
            heatmap,
            width,
            height,
            max_boxes=max_boxes,
            threshold_quantile=threshold_quantile,
        )
        if not candidate_boxes:
            candidate_boxes = heatmap_to_candidate_boxes(heatmap, width, height, max_boxes=max_boxes)
        return LocalizationResult(
            box_xyxy=box,
            candidate_boxes_xyxy=candidate_boxes,
            heatmap=heatmap,
            score=float(heatmap.max()),
            image_embedding=image_embedding[0].astype(np.float32),
        )

    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state[0].detach().cpu().float().numpy()
        patch_embeddings = self._patch_tokens(hidden)
        return patch_embeddings.mean(axis=0).astype(np.float32)

    @staticmethod
    def _patch_tokens(hidden: np.ndarray) -> np.ndarray:
        token_count = hidden.shape[0]
        grid_with_cls = int(np.sqrt(token_count - 1))
        if grid_with_cls * grid_with_cls == token_count - 1:
            return hidden[1:]
        grid_without_cls = int(np.sqrt(token_count))
        if grid_without_cls * grid_without_cls == token_count:
            return hidden
        return hidden[1:]


def normalize(values: np.ndarray) -> np.ndarray:
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def heatmap_to_box(
    heatmap: np.ndarray,
    image_width: int,
    image_height: int,
    threshold_quantile: float,
) -> tuple[int, int, int, int]:
    threshold = float(np.quantile(heatmap, threshold_quantile))
    ys, xs = np.where(heatmap >= threshold)
    if len(xs) == 0 or len(ys) == 0:
        best_y, best_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        xs = np.array([best_x])
        ys = np.array([best_y])

    grid_h, grid_w = heatmap.shape
    x1 = int(xs.min() / grid_w * image_width)
    y1 = int(ys.min() / grid_h * image_height)
    x2 = int((xs.max() + 1) / grid_w * image_width)
    y2 = int((ys.max() + 1) / grid_h * image_height)
    return x1, y1, min(image_width - 1, x2), min(image_height - 1, y2)


def heatmap_to_candidate_boxes(
    heatmap: np.ndarray,
    image_width: int,
    image_height: int,
    max_boxes: int = 8,
    min_distance: int = 2,
) -> list[tuple[int, int, int, int]]:
    grid_h, grid_w = heatmap.shape
    flat_indices = np.argsort(heatmap.reshape(-1))[::-1]
    peaks: list[tuple[int, int]] = []

    for flat_index in flat_indices:
        y, x = np.unravel_index(flat_index, heatmap.shape)
        if any(abs(y - py) <= min_distance and abs(x - px) <= min_distance for py, px in peaks):
            continue
        peaks.append((int(y), int(x)))
        if len(peaks) >= max_boxes:
            break

    cell_w = image_width / grid_w
    cell_h = image_height / grid_h
    boxes = []
    for y, x in peaks:
        x1 = int(max(0, (x - 1) * cell_w))
        y1 = int(max(0, (y - 1) * cell_h))
        x2 = int(min(image_width - 1, (x + 2) * cell_w))
        y2 = int(min(image_height - 1, (y + 2) * cell_h))
        boxes.append((x1, y1, x2, y2))
    return boxes


def heatmap_to_connected_boxes(
    heatmap: np.ndarray,
    image_width: int,
    image_height: int,
    max_boxes: int = 8,
    threshold_quantile: float = 0.82,
    min_cells: int = 2,
) -> list[tuple[int, int, int, int]]:
    threshold = max(float(np.quantile(heatmap, threshold_quantile)), 0.55)
    mask = heatmap > threshold
    components = connected_components(mask)
    grid_h, grid_w = heatmap.shape
    candidates = []

    for component in components:
        if len(component) < min_cells:
            continue
        ys = np.array([cell[0] for cell in component])
        xs = np.array([cell[1] for cell in component])
        score = float(heatmap[ys, xs].mean())
        x1 = int(xs.min() / grid_w * image_width)
        y1 = int(ys.min() / grid_h * image_height)
        x2 = int((xs.max() + 1) / grid_w * image_width)
        y2 = int((ys.max() + 1) / grid_h * image_height)
        box = (x1, y1, min(image_width - 1, x2), min(image_height - 1, y2))
        if box_area(box) < image_width * image_height * 0.005:
            continue
        candidates.append((score, box))

    candidates.sort(key=lambda item: item[0], reverse=True)
    boxes = non_max_suppression([box for _, box in candidates], iou_threshold=0.25)
    return boxes[:max_boxes]


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append(component)
    return components


def non_max_suppression(
    boxes: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if all(iou(box, kept_box) < iou_threshold for kept_box in kept):
            kept.append(box)
    return kept


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)
