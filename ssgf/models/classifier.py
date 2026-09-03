"""Modality classifier -- the CSF baseline "source-identity" head."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config.defaults import Config
from .backbone import _conv_block


class ModalityClassifier(nn.Module):
    """Binary head distinguishing visible vs infrared features.

    Produces a ``(B, 2)`` logit tensor after conv+max-pool layers and global
    average pooling, mirroring ``networks.Classification``.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        ch = cfg.cls_ch
        self.layers = nn.ModuleList()
        for i in range(len(ch) - 1):
            cin, cout = ch[i], ch[i + 1]
            self.layers.append(
                nn.Sequential(_conv_block(cin, cout, cfg.kernel_size, cfg.leaky), nn.MaxPool2d(2))
            )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        out = feats
        for layer in self.layers:
            out = layer(out)
        return torch.mean(out, dim=(2, 3))  # (B, 2)
