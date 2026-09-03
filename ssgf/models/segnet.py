"""Lightweight U-Net semantic segmentation model used for the downstream task."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.b(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 8, base: int = 32):
        super().__init__()
        b0 = self._down(in_ch, base)
        b1 = self._down(base, base * 2)
        b2 = self._down(base * 2, base * 4)
        b3 = self._down(base * 4, base * 8)
        self.b0, self.p0 = b0
        self.b1, self.p1 = b1
        self.b2, self.p2 = b2
        self.b3, self.p3 = b3
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.b4 = _DoubleConv(base * 8, base * 4)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.b5 = _DoubleConv(base * 4, base * 2)
        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.b6 = _DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, n_classes, 1)

    def _down(self, cin, cout):
        return _DoubleConv(cin, cout), nn.MaxPool2d(2)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(self.p0(x0))
        x2 = self.b2(self.p1(x1))
        x3 = self.b3(self.p2(x2))
        x = self.b4(torch.cat([self.up2(x3), x2], 1))
        x = self.b5(torch.cat([self.up1(x), x1], 1))
        x = self.b6(torch.cat([self.up0(x), x0], 1))
        return self.head(x)
