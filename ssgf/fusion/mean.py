"""Mean fusion rule -- the simplest "no saliency" baseline."""

from __future__ import annotations

import torch

from ..config.defaults import Config
from .base import BaseRule


class MeanRule(BaseRule):
    def saliency(self, vis_n: torch.Tensor, ir_n: torch.Tensor):
        ones = torch.ones_like(vis_n[:, :, :, :]).to(self.device)  # (1,1,H,W)
        return ones * 0.5, ones * 0.5
