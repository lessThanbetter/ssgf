"""Fine-tune the fusion decoder on fused features (structural fix).

Motivation: a decoder trained only on single-source features decodes *fused*
feature vectors poorly (out-of-distribution), which caps VIF. Here we keep the
encoder frozen and fine-tune the decoder to map l1-fused features back to the
mean source image, so the decoder learns to handle fused features.

Usage (server):
    python finetune_decoder.py --msrs /root/autodl-tmp/datasets/MSRS-main \
        --ae data/msrs_ae_wide2.pt --out data/msrs_ae_wide3.pt --epochs 10
"""

from __future__ import annotations

import argparse, os, random
import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.models import Encoder, Decoder
from ssgf.models.baselines import fuse_l1
from ssgf.train import ssim_map
from ssgf.utils.checkpoint import load_ckpt, save_ckpt
from ssgf.utils.io import imread, find_pairs


class PairDS(torch.utils.data.Dataset):
    def __init__(self, msrs, patch=128):
        self.pairs = find_pairs(f"{msrs}/train", "vi", "ir")
        self.patch = patch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        vp, ip = self.pairs[i]
        vis = imread(vp); ir = imread(ip)
        h, w = min(vis.shape[0], ir.shape[0]), min(vis.shape[1], ir.shape[1])
        vis, ir = vis[:h, :w], ir[:h, :w]
        p = self.patch
        y = random.randint(0, h - p); x = random.randint(0, w - p)
        vis = vis[y:y+p, x:x+p]; ir = ir[y:y+p, x:x+p]
        vis_n = torch.from_numpy(vis) * 2 - 1   # [-1,1]
        ir_n = torch.from_numpy(ir) * 2 - 1
        target = 0.5 * (vis_n + ir_n)
        return vis_n.unsqueeze(0), ir_n.unsqueeze(0), target.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msrs", required=True)
    ap.add_argument("--ae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config(backbone="wide")
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae, encoder=enc, decoder=dec)
    enc.to(dev).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    dec.to(dev).train()
    ds = PairDS(args.msrs)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    opt = torch.optim.Adam(dec.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        tot = n = 0
        for vn, i_n, target in loader:
            vn, i_n, target = vn.to(dev), i_n.to(dev), target.to(dev)
            ff = fuse_l1(enc(vn), enc(i_n))
            recon = dec(ff)
            loss = 25 * torch.mean((recon - target) ** 2) + (1 - ssim_map(recon, target))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep+1}/{args.epochs} fuse-recon loss={tot/max(n,1):.4f}")
    save_ckpt(args.out, enc, dec, cfg=cfg, meta={"phase": "fused-decoder"})
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
