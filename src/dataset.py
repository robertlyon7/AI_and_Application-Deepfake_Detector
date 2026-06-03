"""PyTorch Datasets for cached face crops from FF++ and Celeb-DF v2."""

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# SBI uses raw [0, 1] inputs — no ImageNet mean/std subtraction.
# These constants are kept only because visualize.py imports them for Grad-CAM
# de-normalization; the values now match a no-op normalization.
IMAGENET_MEAN = [0.0, 0.0, 0.0]
IMAGENET_STD = [1.0, 1.0, 1.0]
SBI_INPUT_SIZE = 380
XCEPTION_INPUT_SIZE = 299


def default_transform() -> Callable:
    """Return the standard SBI eval transform: resize + ToTensor (i.e., /255).

    SBI feeds the network `img.float() / 255` with NO ImageNet normalization,
    so we deliberately do NOT apply transforms.Normalize here.
    """
    return transforms.Compose([
        transforms.Resize((SBI_INPUT_SIZE, SBI_INPUT_SIZE)),
        transforms.ToTensor(),
    ])


def xception_transform() -> Callable:
    """Eval transform for the Cadene/timm `legacy_xception` model.

    Xception was trained with inputs in [-1, 1] (mean=0.5, std=0.5) at 299x299.
    The DeepfakeBench FF++ Xception checkpoint follows the same convention.
    """
    return transforms.Compose([
        transforms.Resize((XCEPTION_INPUT_SIZE, XCEPTION_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class _FrameDataset(Dataset):
    """Base dataset that flattens a per-video manifest into a per-frame list."""

    def __init__(self, manifest_path: str, transform: Optional[Callable] = None):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                f"Run scripts/preprocess.py first to create it."
            )
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        # Flatten (video, frame, label) into a single sample list for __getitem__.
        self.samples = []
        for entry in self.manifest:
            for frame_path in entry["frames"]:
                self.samples.append({
                    "frame": frame_path,
                    "label": int(entry["label"]),
                    "video": entry["video"],
                })

        self.transform = transform if transform is not None else default_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        # Load with PIL for torchvision compatibility.
        img = Image.open(sample["frame"]).convert("RGB")
        tensor = self.transform(img)
        return tensor, sample["label"], sample["video"]


class FFPlusPlusDataset(_FrameDataset):
    """FaceForensics++ frame dataset (reads data/cache/ff++/manifest.json by default)."""

    def __init__(self, manifest_path: str = "data/cache/ff++/manifest.json",
                 transform: Optional[Callable] = None):
        super().__init__(manifest_path, transform)


class CelebDFDataset(_FrameDataset):
    """Celeb-DF v2 frame dataset (reads data/cache/celebdf/manifest.json by default)."""

    def __init__(self, manifest_path: str = "data/cache/celebdf/manifest.json",
                 transform: Optional[Callable] = None):
        super().__init__(manifest_path, transform)
