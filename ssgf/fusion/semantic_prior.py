"""NOVEL semantic-prior fusion rule.

The vanilla CSF rule picks per-pixel weights from a *modality-identity*
classifier. Here the per-pixel decision is driven by a **pretrained semantic
prior** (e.g. salient-object saliency) combined with feature activity, so the
fused features keep whichever modality is more semantically important.

This is the runnable "direction A" trial rule; the same machinery later learns
the prior as a trainable probe (``SemanticSaliencyRule``).
"""

from __future__ import annotations

import numpy as np
import torch

from ..config.defaults import Config
from .base import BaseRule


def _to_np01(x: torch.Tensor) -> np.ndarray:
    """``[-1,1]`` torch ``(1,1,H,W)`` -> ``(H,W)`` numpy in ``[0,1]``."""
    return ((x.detach().cpu().numpy()[0, 0]) + 1.0) / 2.0


class SemanticPriorRule(BaseRule):
    def __init__(self, encoder, decoder, cfg: Config, prior, device: str = "cuda", temperature: float = 0.3):
        super().__init__(encoder, decoder, cfg, device)
        self.prior = prior
        self.temperature = temperature

    def saliency(self, vis_n: torch.Tensor, ir_n: torch.Tensor):
        act_vis = self.encoder(vis_n).abs().sum(dim=1)      # (1,H,W)
        act_ir = self.encoder(ir_n).abs().sum(dim=1)
        s_vis = torch.from_numpy(self.prior(_to_np01(vis_n))).to(self.device).unsqueeze(0)  # (1,H,W)
        s_ir = torch.from_numpy(self.prior(_to_np01(ir_n))).to(self.device).unsqueeze(0)
        d_vis = s_vis * act_vis
        d_ir = s_ir * act_ir
        mx = torch.max(d_vis.max(), d_ir.max()) + 1e-8
        e_vis = torch.exp(d_vis / mx / self.temperature)
        e_ir = torch.exp(d_ir / mx / self.temperature)
        w_vis = e_vis / (e_vis + e_ir)
        w_ir = e_ir / (e_vis + e_ir)
        # broadcast to (1,C,H,W)
        return w_vis.unsqueeze(0), w_ir.unsqueeze(0)
