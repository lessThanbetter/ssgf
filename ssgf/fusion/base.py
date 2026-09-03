"""Shared plumbing for fusion rules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..config.defaults import Config


def to_net_input(img: torch.Tensor) -> torch.Tensor:
    """Map a ``[0, 1]`` image tensor to ``[-1, 1]`` (CSF convention)."""
    return img * 2.0 - 1.0


class BaseRule(ABC):
    """A fusion rule encodes the two sources, computes per-channel saliency
    weights, fuses the feature maps and decodes the result.
    """

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, cfg: Config, device: str = "cuda"):
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.cfg = cfg
        self.device = device
        for m in (self.encoder, self.decoder):
            for p in m.parameters():
                p.requires_grad_(False)

    @abstractmethod
    def saliency(self, vis_n: torch.Tensor, ir_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(w_vis, w_ir)`` with shape ``(1, C, H, W)``."""

    def fuse_pair(self, vis: torch.Tensor, ir: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """vis/ir are ``[0, 1]`` tensors of shape ``(1, 1, H, W)``."""
        vis_n = to_net_input(vis).to(self.device)
        ir_n = to_net_input(ir).to(self.device)

        vis_feats = self.encoder(vis_n)
        ir_feats = self.encoder(ir_n)
        w_vis, w_ir = self.saliency(vis_n, ir_n)

        fuse_feats = w_vis * vis_feats + w_ir * ir_feats
        fused_n = self.decoder(fuse_feats)          # (1, 1, H, W) in [-1, 1]
        fused = (fused_n + 1.0) / 2.0               # back to [0, 1]
        debug = {
            "w_vis": w_vis.detach(),
            "w_ir": w_ir.detach(),
            "vis_feats": vis_feats.detach(),
            "ir_feats": ir_feats.detach(),
            "fused_n": fused_n.detach(),
        }
        return fused.clamp(0, 1), debug

    def _normalize(self, d_vis: torch.Tensor, d_ir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-offset + per-channel softmax with temperature (as in CSF)."""
        if self.cfg.mean_offset:
            d_ir = d_ir + (d_vis.mean() - d_ir.mean())
        e_vis = torch.exp(d_vis / self.cfg.temperature)
        e_ir = torch.exp(d_ir / self.cfg.temperature)
        w_vis = e_vis / (e_vis + e_ir)
        w_ir = e_ir / (e_vis + e_ir)
        return w_vis.unsqueeze(0), w_ir.unsqueeze(0)
