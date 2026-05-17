from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


@dataclass
class VisionEncoderConfig:
    """Configuration for a Hugging Face vision encoder."""

    model_name: str = "facebook/dinov2-base"
    device: str | None = None
    batch_size: int = 8


class VisionEmbeddingAdapter:
    """Adapter layer for JEPA-style image encoders.

    Today this wraps Hugging Face vision backbones such as DINOv2. Future
    I-JEPA or V-JEPA implementations can keep the same `embed_images` surface.
    """

    def __init__(self, config: VisionEncoderConfig):
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def embed_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        pil_images = [ensure_rgb(image) for image in images]
        if not pil_images:
            return np.empty((0, 0), dtype=np.float32)

        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(pil_images), self.config.batch_size):
                batch = pil_images[start : start + self.config.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                pooled = self._pool_outputs(outputs)
                batches.append(pooled.detach().cpu().float().numpy())
        return np.vstack(batches).astype(np.float32)

    @staticmethod
    def _pool_outputs(outputs) -> torch.Tensor:
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        if getattr(outputs, "last_hidden_state", None) is None:
            raise ValueError("Model output does not include pooler_output or last_hidden_state.")
        return outputs.last_hidden_state.mean(dim=1)


def ensure_rgb(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}.")
    if image.mode != "RGB":
        return image.convert("RGB")
    return image
