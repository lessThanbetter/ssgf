"""NOVEL semantic saliency rule -- saliency from a *learned* semantic probe.

The probe is trained to predict a semantic-importance map from the encoder
features. At fusion time it outputs that map directly for each source; the map is
multiplied by the feature activity and combined into per-pixel source weights.
This keeps semantically important content rather than modality-statistical
outliers, and is the *learned* counterpart of the off-the-shelf ``semantic_prior``
rule (direction A trial).
"""

from __future__ import annotations

import numpy as np
import torch

from ..config.defaults import Config
from ..models import build_probe
from .base import BaseRule


def _to_np01(x: torch.Tensor) -> np.ndarray:
    """``[-1,1]`` torch ``(1,1,H,W)`` -> ``(H,W)`` numpy in ``[0,1]``."""
    return ((x.detach().cpu().numpy()[0, 0]) + 1.0) / 2.0


class SemanticSaliencyRule(BaseRule):
    def __init__(self, encoder, decoder, cfg: Config, device: str = "cuda",
                 n_classes: int = 1, temperature: float = 0.3, prior=None):
        super().__init__(encoder, decoder, cfg, device)
        self.probe = build_probe(cfg).to(device).eval()
        for p in self.probe.parameters():
            p.requires_grad_(False)
        self.temperature = temperature
        self.prior = prior  # optional classical prior for a robust hybrid

    def saliency(self, vis_n: torch.Tensor, ir_n: torch.Tensor):
        vis_feats = self.encoder(vis_n)
        ir_feats = self.encoder(ir_n)
        # Learned semantic importance map per source (probe predicts [0,1]).
        s_vis = torch.sigmoid(self.probe.map_logits(vis_feats))   # (1,1,H,W)
        s_ir = torch.sigmoid(self.probe.map_logits(ir_feats))
        if self.prior is not None:
            p_vis = torch.from_numpy(self.prior(_to_np01(vis_n))).to(self.device).unsqueeze(0).unsqueeze(0)
            p_ir = torch.from_numpy(self.prior(_to_np01(ir_n))).to(self.device).unsqueeze(0).unsqueeze(0)
            # Hybrid prior: never drop a target that the classical prior detects.
            s_vis = torch.maximum(s_vis, p_vis)
            s_ir = torch.maximum(s_ir, p_ir)
        act_vis = vis_feats.abs().sum(dim=1, keepdim=True)        # (1,1,H,W)
        act_ir = ir_feats.abs().sum(dim=1, keepdim=True)
        d_vis = s_vis * act_vis
        d_ir = s_ir * act_ir
        mx = torch.max(d_vis.max(), d_ir.max()) + 1e-8
        e_vis = torch.exp(d_vis / mx / self.temperature)
        e_ir = torch.exp(d_ir / mx / self.temperature)
        w_vis = e_vis / (e_vis + e_ir)
        w_ir = e_ir / (e_vis + e_ir)
        return w_vis, w_ir
