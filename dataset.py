import os
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image
import skimage.color as skcolor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset


_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_image_paths(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")
    paths = [p for p in root_path.rglob("*") if p.suffix.lower() in _VALID_EXTENSIONS]
    if not paths:
        raise RuntimeError(f"No images found in {root}")
    return sorted(paths)


def _build_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
    ])


def _build_val_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.CenterCrop(height=image_size, width=image_size),
    ])


class ColorizationDataset(Dataset):
    """Loads color images and returns (L_tensor, ab_tensor) in [-1, 1].

    L channel range [0, 100] is mapped to [-1, 1].
    ab channels range [-128, 127] are mapped to [-1, 1].
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 256,
        val_fraction: float = 0.1,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        all_paths = _collect_image_paths(data_dir)
        n_val = max(1, int(len(all_paths) * val_fraction))
        if split == "val":
            self.paths = all_paths[:n_val]
            self.transform = _build_val_transforms(image_size)
        else:
            self.paths = all_paths[n_val:]
            self.transform = _build_train_transforms(image_size)

        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def _load_lab(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0
        lab = skcolor.rgb2lab(img_np).astype(np.float32)
        return lab

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lab = self._load_lab(self.paths[idx])

        # Albumentations expects uint8 or float32 HWC; we pass float32 HWC
        augmented = self.transform(image=lab)
        lab = augmented["image"]  # (H, W, 3)

        L = lab[:, :, 0:1]   # (H, W, 1), range [0, 100]
        ab = lab[:, :, 1:3]  # (H, W, 2), range [-128, 127]

        # Normalize to [-1, 1]
        L_norm = (L / 50.0) - 1.0
        ab_norm = ab / 110.0

        L_tensor = torch.from_numpy(L_norm).permute(2, 0, 1)   # (1, H, W)
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1)  # (2, H, W)

        return L_tensor, ab_tensor
