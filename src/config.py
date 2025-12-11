from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from dotenv import load_dotenv

load_dotenv()

def determine_device():
    """
    Pick the best available device:
      - CUDA, if available
      - MPS (Apple), if available
      - CPU otherwise
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return "cpu"


@dataclass
class AppConfig:
    text_embed_model_name: str = field(
        default_factory=lambda: os.getenv(
            "TEXT_EMBED_MODEL",
            "intfloat/multilingual-e5-base",
        )
    )
    clip_model_name: str = field(
        default_factory=lambda: os.getenv(
            "CLIP_MODEL",
            "openai/clip-vit-base-patch32",
        )
    )

    chroma_path: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_PATH",
            "data/chroma",
        )
    )

    device: str | torch.device = determine_device()

    google_ai_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_AI_API_KEY", "")
    )
    gemini_model_name: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )


config = AppConfig()
