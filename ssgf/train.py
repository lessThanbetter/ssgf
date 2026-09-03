"""Training entry point.

Mode ``ae`` trains the auto-encoder (self-supervised reconstruction) -- this is
the first thing to run after downloading the VIF training images, since it gives
the encoder/decoder that ``fuse.py`` depends on.

Mode ``probe`` trains the semantic probe on a set of foreground/importance masks
(one mask per source image). This is what makes the *semantic* saliency rule
meaningful. E.g.
    python -m ssgf.train --mode ae --root /data/raw --out /data/ssgf/ae.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import get_default_config
from .data import FusionDataset
from .models import Decoder, Encoder, build_probe
from .utils.checkpoint import load_ckpt, save_ckpt
from .utils.io import find_pairs, imread


def _gauss_window(size: int, sigma: float, device):
    x = torch.arange(size, device=device).float() - (size // 2)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = torch.outer(g, g)
    k = k / k.sum()
    return k.view(1, 1, size, size)


def ssim_map(img1: torch.Tensor, img2: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    window = _gauss_window(size, sigma, img1.device)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = size // 2
    mu1 = F.conv2d(img1, window, padding=pad)
    mu2 = F.conv2d(img2, window, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=pad) - mu1_mu2
    v = ((2 * s12 + C2) * (2 * mu1_mu2 + C1)) / ((s1 + s2 + C2) * (mu1_sq + mu2_sq + C1))
    return v.mean()


def train_ae(cfg, encoder, decoder, root, vis_dir, ir_dir, out_path) -> None:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    encoder.to(device).train()
    decoder.to(device).train()
    ds = FusionDataset(root, vis_dir, ir_dir, patch_size=cfg.patch_size, train=True)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.decay_rate)

    print(f"[ae] {len(ds)} source pairs -> {len(loader)} batches, device={device}")
    for epoch in range(cfg.epochs):
        tot, n = 0.0, 0
        for batch in loader:
            x = batch.to(device)              # (B,1,P,P) in [0,1]
            xn = x * 2 - 1                     # [-1,1]
            recon = decoder(encoder(xn))
            loss = cfg.recon_mse_weight * F.mse_loss(xn, recon) + (1 - ssim_map(xn, recon))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item(); n += 1
            for p in params:
                p.data.clamp_(-50, 50)
        scheduler.step()
        print(f"epoch {epoch+1}/{cfg.epochs} loss={tot/n:.4f}")
        save_ckpt(out_path, encoder, decoder, cfg=cfg, meta={"phase": "ae"})
    print(f"[ae] saved -> {out_path}")


class MaskDataset(torch.utils.data.Dataset):
    """(source image, semantic mask) pairs for probe distillation.

    Each sample randomly takes either the visible or the infrared source and its
    corresponding teacher mask (from ``U2Net`` pseudo-labels), so the probe learns
    to map either modality's features to a semantic-importance map.
    """

    def __init__(self, root, mask_vis_dir, mask_ir_dir, vis_dir="vis", ir_dir="ir", patch_size=128):
        self.pairs = find_pairs(root, vis_dir, ir_dir)
        self.mask_vis_dir = mask_vis_dir
        self.mask_ir_dir = mask_ir_dir
        self.patch_size = patch_size

    def __len__(self):
        return len(self.pairs) * 2  # both modalities

    def __getitem__(self, idx):
        pair_idx = idx // 2
        modality = idx % 2  # 0=vis, 1=ir
        vis_path, ir_path = self.pairs[pair_idx]
        if modality == 0:
            src_path, mask_root = vis_path, self.mask_vis_dir
        else:
            src_path, mask_root = ir_path, self.mask_ir_dir
        name = os.path.splitext(os.path.basename(src_path))[0]
        src = imread(src_path)
        mask = imread(os.path.join(mask_root, name + ".png"))
        h, w = min(src.shape[0], mask.shape[0]), min(src.shape[1], mask.shape[1])
        src, mask = src[:h, :w], mask[:h, :w]
        p = self.patch_size
        if h >= p and w >= p:
            y = np.random.randint(0, h - p + 1)
            x = np.random.randint(0, w - p + 1)
            src = src[y:y + p, x:x + p]
            mask = mask[y:y + p, x:x + p]
        return torch.from_numpy(src).unsqueeze(0), torch.from_numpy(np.clip(mask, 0, 1)).unsqueeze(0)


def train_probe(cfg, encoder, decoder, probe, root, vis_dir, ir_dir, mask_vis_dir, mask_ir_dir,
                ae_ckpt, out_path) -> None:
    if not (os.path.isdir(mask_vis_dir) and os.path.isdir(mask_ir_dir)):
        raise SystemExit(f"[probe] mask dirs not found: {mask_vis_dir}, {mask_ir_dir}")
    device = cfg.device if torch.cuda.is_available() else "cpu"
    load_ckpt(ae_ckpt, encoder=encoder, decoder=decoder)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    probe.to(device).train()
    ds = MaskDataset(root, mask_vis_dir, mask_ir_dir, vis_dir, ir_dir, patch_size=cfg.patch_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    optimizer = torch.optim.Adam(probe.parameters(), lr=cfg.lr)
    print(f"[probe] {len(ds)} masked pairs, device={device}")
    for epoch in range(cfg.epochs):
        tot, n = 0.0, 0
        for src, mask in loader:
            src, mask = src.to(device), mask.to(device)
            feats = encoder(src * 2 - 1)
            logits = probe.map_logits(feats)          # (B,1,H,W)
            loss = F.binary_cross_entropy_with_logits(logits, mask)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item(); n += 1
        print(f"epoch {epoch+1}/{cfg.epochs} bce={tot/n:.4f}")
        save_ckpt(out_path, encoder, decoder, probe=probe, cfg=cfg, meta={"phase": "probe"})
    print(f"[probe] saved -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SSGF training")
    ap.add_argument("--mode", default="ae", choices=["ae", "probe"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--vis_dir", default="vis")
    ap.add_argument("--ir_dir", default="ir")
    ap.add_argument("--mask_dir", default=None, help="mask folder for --mode probe")
    ap.add_argument("--mask_vis_dir", default=None, help="mask folder for VIS (--mode probe)")
    ap.add_argument("--mask_ir_dir", default=None, help="mask folder for IR (--mode probe)")
    ap.add_argument("--ae_ckpt", default=None, help="encoder/decoder checkpoint to init --mode probe")
    ap.add_argument("--out", required=True, help="checkpoint output path (.pt)")
    ap.add_argument("--cfg", nargs="*", default=[], help="key=value config override")
    args = ap.parse_args()

    overrides = _parse_cfg(args.cfg)
    cfg = get_default_config(**overrides)
    seed = cfg.seed
    torch.manual_seed(seed); np.random.seed(seed)

    encoder, decoder = Encoder(cfg), Decoder(cfg)
    if args.mode == "ae":
        train_ae(cfg, encoder, decoder, args.root, args.vis_dir, args.ir_dir, args.out)
    else:
        probe = build_probe(cfg)
        train_probe(cfg, encoder, decoder, probe, args.root, args.vis_dir, args.ir_dir,
                    args.mask_vis_dir or args.mask_dir, args.mask_ir_dir or args.mask_dir,
                    args.ae_ckpt, args.out)


def _parse_cfg(pairs: list[str]) -> dict:
    out = {}
    for kv in pairs:
        k, _, v = kv.partition("=")
        try:
            out[k.strip()] = int(v.strip())
        except ValueError:
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                out[k.strip()] = v.strip()
    return out


if __name__ == "__main__":
    main()
