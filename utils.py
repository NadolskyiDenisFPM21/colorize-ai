import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import skimage.color as skcolor


class AverageMeter:
    """Tracks a running average of a scalar value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str,
) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> int:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)


def lab_to_rgb(L: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    """Convert normalized L and ab tensors to an RGB numpy array (H, W, 3) uint8.

    Args:
        L:  (1, H, W) or (H, W) tensor in [-1, 1]
        ab: (2, H, W) or (H, W, 2) tensor in [-1, 1]

    Returns:
        RGB image as uint8 numpy array (H, W, 3).
    """
    if L.dim() == 3:
        L = L.squeeze(0)
    L_np = ((L.detach().cpu().float() + 1.0) * 50.0).numpy()

    if ab.dim() == 3:
        ab_np = ab.detach().cpu().float().permute(1, 2, 0).numpy()
    else:
        ab_np = ab.detach().cpu().float().numpy()
    ab_np = ab_np * 110.0

    lab = np.concatenate([L_np[:, :, np.newaxis], ab_np], axis=2).astype(np.float32)
    rgb = skcolor.lab2rgb(lab)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
