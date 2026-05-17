from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image


DegradationMode = Literal["none", "pixel", "patch", "both"]
FillMode = Literal["black", "gray", "noise"]


@dataclass(frozen=True)
class DegradationConfig:
    mode: DegradationMode = "none"
    ratio: float = 0.0
    patch_size: int = 32
    fill: FillMode = "gray"
    seed: int = 7


def degrade_image(image: Image.Image, config: DegradationConfig, sample_index: int = 0) -> Image.Image:
    if config.mode == "none" or config.ratio <= 0:
        return image.convert("RGB")
    if not 0 <= config.ratio <= 1:
        raise ValueError("Degradation ratio must be in the range [0, 1].")

    rng = np.random.default_rng(config.seed + sample_index)
    degraded = np.asarray(image.convert("RGB")).copy()

    if config.mode in {"pixel", "both"}:
        degraded = random_pixel_mask(degraded, config.ratio, config.fill, rng)
    if config.mode in {"patch", "both"}:
        degraded = random_patch_mask(degraded, config.ratio, config.patch_size, config.fill, rng)

    return Image.fromarray(degraded.astype(np.uint8), mode="RGB")


def random_pixel_mask(
    image: np.ndarray,
    ratio: float,
    fill: FillMode,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width, _ = image.shape
    mask = rng.random((height, width)) < ratio
    image[mask] = fill_values(mask.sum(), fill, rng)
    return image


def random_patch_mask(
    image: np.ndarray,
    ratio: float,
    patch_size: int,
    fill: FillMode,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width, _ = image.shape
    patch_size = max(1, int(patch_size))
    total_area = height * width
    target_area = int(total_area * ratio)
    masked_area = 0

    while masked_area < target_area:
        patch_w = min(patch_size, width)
        patch_h = min(patch_size, height)
        x = int(rng.integers(0, max(1, width - patch_w + 1)))
        y = int(rng.integers(0, max(1, height - patch_h + 1)))
        image[y : y + patch_h, x : x + patch_w] = fill_values(
            patch_w * patch_h,
            fill,
            rng,
        ).reshape(patch_h, patch_w, 3)
        masked_area += patch_w * patch_h
    return image


def fill_values(count: int, fill: FillMode, rng: np.random.Generator) -> np.ndarray:
    if fill == "black":
        return np.zeros((count, 3), dtype=np.uint8)
    if fill == "noise":
        return rng.integers(0, 256, size=(count, 3), dtype=np.uint8)
    return np.full((count, 3), 127, dtype=np.uint8)
