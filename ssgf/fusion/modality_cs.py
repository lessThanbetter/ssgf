"""CSF baseline rule -- saliency from a *modality-identity* classifier."""

from __future__ import annotations

import torch

from ..config.defaults import Config
from ..models import ModalityClassifier
from .base import BaseRule
from .saliency import interpolation_sequences, path_attribution


class ModalityCSRule(BaseRule):
    """Reproduce the original classification-saliency fusion rule.

    Two paths are evaluated: visible-addition vs infrared-addition. Their
    importance is the gradient of the corresponding class logit w.r.t. the
    feature volume, giving a per-channel weighting that modulates the fusion.
    """

    def __init__(self, encoder, decoder, cfg: Config, device: str = "cuda"):
        super().__init__(encoder, decoder, cfg, device)
        self.scorer = ModalityClassifier(cfg).to(device).eval()
        for p in self.scorer.parameters():
            p.requires_grad_(False)

    def _score(self, feats):
        return self.scorer(feats)

    def saliency(self, vis_n: torch.Tensor, ir_n: torch.Tensor):
        var_ir, var_vis = interpolation_sequences(vis_n, ir_n, self.cfg.samples)
        d_ir = path_attribution(self.encoder, self._score, var_ir, self.cfg.samples, self.cfg.decay, target_idx=1)
        d_vis = path_attribution(self.encoder, self._score, var_vis, self.cfg.samples, self.cfg.decay, target_idx=0)
        return self._normalize(d_vis, d_ir)
