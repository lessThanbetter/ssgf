"""Datasets for auto-encoder reconstruction and fusion inference."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import find_pairs, imread


class FusionDataset(Dataset):
    """Self-supervised reconstruction dataset over VIS/IR source patches.

    In ``recon`` mode each sample is a random ``patch_size`` crop of either the
    visible or the infrared image (mirroring the original CSF mixing strategy).
    In ``pair`` mode the full aligned pair is returned (for inference).
    """

    def __init__(self, root: str, vis_dir: str = "vis", ir_dir: str = "ir",
                 patch_size: int = 128, train: bool = True):
        self.pairs = find_pairs(root, vis_dir, ir_dir)
        self.patch_size = patch_size
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        p = self.patch_size
        if h < p or w < p:
            img = np.pad(img, ((0, max(0, p - h)), (0, max(0, p - w))), mode="edge")
            h, w = img.shape
        y = np.random.randint(0, h - p + 1)
        x = np.random.randint(0, w - p + 1)
        return img[y:y + p, x:x + p]

    def __getitem__(self, idx: int):
        vis_path, ir_path = self.pairs[idx]
        vis = imread(vis_path)
        ir = imread(ir_path)
        if self.train:
            # Randomly pick a source modality per sample (CSF mixing strategy).
            src = vis if np.random.rand() < 0.5 else ir
            patch = self._crop(src)
            return torch.from_numpy(patch).unsqueeze(0)  # (1, P, P)
        vis_p = torch.from_numpy(vis).unsqueeze(0)      # (1, H, W)
        ir_p = torch.from_numpy(ir).unsqueeze(0)
        return vis_p, ir_p
