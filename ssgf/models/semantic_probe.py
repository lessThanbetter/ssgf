"""Semantic probe -- the novel saliency source.

Whereas the CSF baseline uses a modality-identity classifier, this probe
produces a semantic signal (e.g. salient-object / class prior) that measures how
important each pixel is for a downstream high-level task. The attribution of
this semantic score w.r.t. the feature maps gives the saliency.

The head reduces the input feature volume to a foreground logit map and then a
scalar score, which is what we differentiate when computing saliency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config.defaults import Config
from .backbone import _conv_block


class SemanticProbe(nn.Module):
    """Tiny head producing a scalar semantic score from input features.

    ``forward`` returns a ``(B, n_classes)`` tensor (default 1 class). Internally
    it predicts a per-pixel logit map (via ``map_logits``), then averages over
    space. ``map_logits`` exposes the spatial logits for masked supervision.
    """

    def __init__(self, cfg: Config, n_classes: int = 1):
        super().__init__()
        self.cfg = cfg
        ch = list(cfg.probe_ch)
        ch[-1] = n_classes
        self.layers = nn.ModuleList(
            [
                _conv_block(ch[i], ch[i + 1], cfg.probe_kernel, cfg.leaky, use_act=(i < len(ch) - 2))
                for i in range(len(ch) - 1)
            ]
        )

    def map_logits(self, feats: torch.Tensor) -> torch.Tensor:
        out = feats
        for layer in self.layers:
            out = layer(out)
        return out  # (B, n_classes, H, W)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.map_logits(feats), dim=(2, 3))  # (B, n_classes)


class MultiScaleProbe(nn.Module):
    """Multi-scale targetness probe.

    Predicts the targetness logit at the native feature resolution and after a
    spatial pooling (a coarser view), then combines them. This enlarges the
    receptive field of the probe and yields a more robust targetness prior, which
    increases the advantage of the learned probe over the single-scale classical
    top-hat prior.
    """

    def __init__(self, cfg: Config, n_classes: int = 1, scale: int = 2):
        super().__init__()
        self.cfg = cfg
        ch = list(cfg.probe_ch)
        ch[-1] = n_classes
        self.scale = scale
        self.native = nn.ModuleList(
            [_conv_block(ch[i], ch[i + 1], cfg.probe_kernel, cfg.leaky, use_act=(i < len(ch) - 2))
             for i in range(len(ch) - 1)]
        )
        # coarse branch computes on pooled features
        coarse_ch = [int(c) for c in ch]
        self.coarse = nn.ModuleList(
            [_conv_block(coarse_ch[i], coarse_ch[i + 1], cfg.probe_kernel, cfg.leaky, use_act=(i < len(coarse_ch) - 2))
             for i in range(len(coarse_ch) - 1)]
        )

    def map_logits(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats
        for layer in self.native:
            x = layer(x)
        native = x
        pooled = torch.nn.functional.avg_pool2d(feats, self.scale, self.scale)
        y = pooled
        for layer in self.coarse:
            y = layer(y)
        coarse = torch.nn.functional.interpolate(y, size=native.shape[-2:], mode="bilinear", align_corners=False)
        return native + 0.5 * coarse

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.map_logits(feats), dim=(2, 3))


def build_probe(cfg: Config):
    if getattr(cfg, "probe_type", "single") == "multiscale":
        return MultiScaleProbe(cfg)
    return SemanticProbe(cfg)
