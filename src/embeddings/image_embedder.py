from __future__ import annotations

from io import BytesIO
from typing import List, Union
from urllib.parse import urlparse

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel


class ImageEmbedder:
    """
    Wrapper for CLIP HF models that provide:
      - model.get_image_features(...)
      - model.get_text_features(...)

    Works with:
      - "openai/clip-vit-base-patch32"
      - "openai/clip-vit-base-patch16"

    Designed for:
      - movie posters (local paths)
      - short text queries describing posters (for image search).
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def is_url(self, path: str) -> bool:
        try:
            parsed = urlparse(path)
            return parsed.scheme in ("http", "https")
        except Exception:
            return False

    def load_image_from_url(self, url: str) -> Image.Image:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def to_pil(self, image_source) -> Image.Image:
        """
        image_source:
          - str: either URL or local path
          - PIL.Image.Image
        """
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")

        if isinstance(image_source, str) and self.is_url(image_source):
            return self.load_image_from_url(image_source)

        return Image.open(image_source).convert("RGB")

    def _ensure_text_list(self, texts: Union[str, List[str]]) -> List[str]:
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    @torch.no_grad()
    def embed_images(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        """
        Returns a (B, D) tensor of L2-normalized image embeddings.
        """
        pil_images = [self.to_pil(img) for img in images]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    @torch.no_grad()
    def embed_texts(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Returns a (B, D) tensor of L2-normalized text embeddings (CLIP text space).
        """
        text_list = self._ensure_text_list(texts)
        inputs = self.processor(text=text_list, padding=True, return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    def embed_images_to_list(self, images: List[Union[str, Image.Image]]) -> List[List[float]]:
        return self.embed_images(images).cpu().tolist()

    def embed_texts_to_list(self, texts: Union[str, List[str]]) -> List[List[float]]:
        return self.embed_texts(texts).cpu().tolist()
