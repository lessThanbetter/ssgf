"""Train the CSF modality classifier (visible vs infrared) as a fusion baseline.

The original Classification-Saliency rule uses a classifier that distinguishes
visible from infrared features and fuses by gradient saliency. Here we train the
classifier on MSRS features (encoder frozen) and save it with the encoder/decoder
so ``--rule modality_cs`` can reproduce the CSF baseline for novelty comparison.

Usage:
    python train_classifier.py --msrs .../MSRS-main --ae data/msrs_ae.pt \
        --out data/msrs_csf.pt --epochs 20
"""

from __future__ import annotations

import argparse, os, random
import numpy as np
import torch
import torch.nn.functional as F

from ssgf.config import get_default_config
from ssgf.models import Encoder, Decoder, ModalityClassifier
from ssgf.utils.checkpoint import load_ckpt, save_ckpt
from ssgf.utils.io import imread, find_pairs


class ClsDS(torch.utils.data.Dataset):
    def __init__(self, msrs, patch=128):
        self.pairs = find_pairs(f"{msrs}/train", "vi", "ir")
        self.patch = patch

    def __len__(self):
        return len(self.pairs) * 2

    def __getitem__(self, i):
        pi, mod = divmod(i, 2)  # 0=vis, 1=ir
        vp, ip = self.pairs[pi]
        path = vp if mod == 0 else ip
        img = imread(path)
        h, w = img.shape; p = self.patch
        y = random.randint(0, h - p); x = random.randint(0, w - p)
        img = img[y:y+p, x:x+p]
        x = (torch.from_numpy(img) * 2 - 1).unsqueeze(0)  # [-1,1]
        return x, torch.tensor([mod], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msrs", required=True)
    ap.add_argument("--ae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae, encoder=enc, decoder=dec)
    enc.to(dev).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    cls = ModalityClassifier(cfg).to(dev).train()
    ds = ClsDS(args.msrs); loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    opt = torch.optim.Adam(cls.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        tot = n = acc = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            feats = enc(x)
            logits = cls(feats)
            y = y.squeeze(1)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            acc += (logits.argmax(1) == y).float().sum().item()
            tot += loss.item(); n += 1
        print(f"epoch {ep+1}/{args.epochs} bce={tot/max(n,1):.4f} acc={acc/(n*16):.4f}")
    save_ckpt(args.out, enc, dec, classifier=cls, cfg=cfg, meta={"phase": "csf-classifier"})
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
