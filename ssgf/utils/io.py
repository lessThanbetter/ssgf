"""Image read/write helpers (grayscale, float in [0, 1])."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from PIL import Image


def imread(path: str) -> np.ndarray:
    """Read an image as a float array in ``[0, 1]`` with shape ``(H, W)``."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def imsave(path: str, arr: np.ndarray) -> None:
    """Save a float array. Values are assumed in ``[0, 1]`` (clipped)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def find_pairs(root: str, vis_dir: str = "vis", ir_dir: str = "ir") -> list[tuple[str, str]]:
    """Return a list of ``(vis_path, ir_path)`` with matching basenames."""
    vis_root = os.path.join(root, vis_dir)
    ir_root = os.path.join(root, ir_dir)
    vis_files = {os.path.splitext(f)[0]: f for f in os.listdir(vis_root)}
    ir_files = {os.path.splitext(f)[0]: f for f in os.listdir(ir_root)}
    keys = sorted(set(vis_files) & set(ir_files))
    return [(os.path.join(vis_root, vis_files[k]), os.path.join(ir_root, ir_files[k])) for k in keys]
