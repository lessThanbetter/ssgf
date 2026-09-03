"""Encoder / Decoder.

Faithful PyTorch port of ``networks.py`` from the CSF code: reflection padding,
leaky-ReLU, stride-1 convs. The encoder preserves spatial resolution and
outputs ``enc_ch[-1]`` (default 24) channels; the decoder reconstructs a single
channel in ``[-1, 1]`` via a tanh at the final layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config.defaults import Config


def _conv_block(cin: int, cout: int, k: int, leaky: float, use_act: bool = True) -> nn.Module:
    layers: list[nn.Module] = [nn.ReflectionPad2d(k // 2), nn.Conv2d(cin, cout, k, stride=1, bias=True)]
    if use_act:
        layers.append(nn.LeakyReLU(leaky, inplace=True))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        ch = cfg.enc_ch
        self.layers = nn.ModuleList(
            [_conv_block(ch[i], ch[i + 1], cfg.kernel_size, cfg.leaky) for i in range(len(ch) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Decoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        ch = cfg.dec_ch
        self.layers = nn.ModuleList(
            [
                _conv_block(ch[i], ch[i + 1], cfg.kernel_size, cfg.leaky, use_act=(i < len(ch) - 2))
                for i in range(len(ch) - 1)
            ]
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        out = feats
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i == n - 1:
                if getattr(self.cfg, "decoder_tanh", True):
                    out = torch.tanh(out)
        return out
