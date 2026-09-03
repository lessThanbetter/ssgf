"""IFCNN and DenseFuse baseline networks (trained on the same data for fairness)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IFCNN(nn.Module):
    """IFCNN (Zhang et al., Inf. Fusion 2020): 2-layer extractor + reconstructor."""

    def __init__(self, in_ch: int = 1, base: int = 64):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(base, base, 3, padding=1), nn.LeakyReLU(0.2),
        )
        self.reconstructor = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def encode(self, x):
        return self.extractor(x)

    def decode(self, f):
        return self.reconstructor(f)


class _Dense(nn.Module):
    def __init__(self, in_ch, growth=8):
        super().__init__()
        self.c = nn.Conv2d(in_ch, growth, 3, padding=1)
        self.l = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.l(self.c(x))
        return torch.cat([x, y], 1)


class DenseFuse(nn.Module):
    """DenseFuse (Li et al., Inf. Fusion 2021) encoder (dense blocks) + decoder."""

    def __init__(self, in_ch: int = 1, base: int = 16, growth: int = 8, nblocks: int = 4):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.LeakyReLU(0.2))
        self.blocks = nn.ModuleList()
        cin = base
        for _ in range(nblocks):
            self.blocks.append(_Dense(cin, growth))
            cin += growth
        self.decoder = nn.Sequential(
            nn.Conv2d(cin, 64, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, in_ch, 3, padding=1),
        )

    def encode(self, x):
        f = self.head(x)
        for b in self.blocks:
            f = b(f)
        return f

    def decode(self, f):
        return self.decoder(f)


def l1_activity(features):
    """Per-pixel activity = L1-norm over channels."""
    return features.abs().sum(dim=1, keepdim=True)


def fuse_l1(feats_a, feats_b):
    """L1-norm fusion: per-pixel weighted average by activity."""
    a, b = l1_activity(feats_a), l1_activity(feats_b)
    w = a / (a + b + 1e-8)
    return w * feats_a + (1 - w) * feats_b


def fuse_mean(feats_a, feats_b):
    return 0.5 * feats_a + 0.5 * feats_b


def fuse_max(feats_a, feats_b):
    return torch.maximum(feats_a, feats_b)
