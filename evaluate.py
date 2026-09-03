"""Reproducible cross-dataset evaluation for the SSGF rules.

Computes mean / classical target prior (top-hat) / learned semantic probe /
ensemble on a set of VIS-IR datasets and writes mean+/-std metrics to CSV.

Usage (on the server):
    python evaluate.py --ae data/msrs_ae.pt --probe data/msrs_probe.pt \
        --datasets TNO|data/tno_demo|vis|ir \
        --datasets MSRS|.../MSRS-main/test|vi|ir \
        --datasets RoadScene|.../RoadScene-master|crop_HR_visible|infrared \
        --out data/cross_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, SemanticProbe, build_prior
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import find_pairs, imread
from ssgf.utils.metrics import report


def _crop(a, b):
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def _resize_pair(a, b, max_side):
    if max_side <= 0:
        return a, b
    h, w = a.shape
    scale = max_side / max(h, w)
    if scale >= 1:
        return a, b
    nw, nh = int(w * scale), int(h * scale)
    def rz(x):
        return np.asarray(Image.fromarray((x * 255).astype("uint8")).resize((nw, nh)), dtype=np.float32) / 255.0
    return rz(a), rz(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", required=True, help="encoder/decoder checkpoint")
    ap.add_argument("--probe", required=True, help="learned-probe checkpoint")
    ap.add_argument("--prior", default="tophat", help="classical prior spec (default tophat)")
    ap.add_argument("--datasets", action="append", required=True,
                    help="name::root::vis_dir::ir_dir (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="per-dataset cap (0 = all)")
    ap.add_argument("--max_side", type=int, default=0, help="cap longest side (0 = native)")
    ap.add_argument("--out", default="data/cross_dataset.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe_type", default="single", choices=["single", "multiscale"])
    ap.add_argument("--backbone", default="csf", choices=["csf", "wide"])
    args = ap.parse_args()

    cfg = get_default_config(probe_type=args.probe_type, backbone=args.backbone)
    device = args.device if torch.cuda.is_available() else "cpu"
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae, encoder=enc, decoder=dec)

    prior = build_prior(args.prior)
    mean_rule = MeanRule(enc, dec, cfg, device)
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, device)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, device)
    load_ckpt(args.probe, probe=probe_rule.probe)   # must load into the rule's probe

    methods = ["mean", "tophat", "probe", "ensemble"]
    parsed = []
    for spec in args.datasets:
        name, root, vd, idd = spec.split("::")
        pairs = find_pairs(root, vd, idd)
        if args.limit > 0:
            pairs = pairs[: args.limit]
        parsed.append((name, pairs))
        print(f"[eval] {name}: {len(pairs)} pairs")

    results = {m: {name: [] for name, _ in parsed} for m in methods}

    def fuse(rule, vis, ir):
        v = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0)
        i = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0)
        f, _ = rule.fuse_pair(v, i)
        return f.squeeze().cpu().numpy()

    for name, pairs in parsed:
        for vis_path, ir_path in pairs:
            vis, ir = _crop(imread(vis_path), imread(ir_path))
            vis, ir = _resize_pair(vis, ir, args.max_side)
            f_mean = fuse(mean_rule, vis, ir)
            f_top = fuse(top_rule, vis, ir)
            f_probe = fuse(probe_rule, vis, ir)
            f_ens = (f_top + f_probe) / 2.0
            for m, f in [("mean", f_mean), ("tophat", f_top), ("probe", f_probe), ("ensemble", f_ens)]:
                r = report(f, vis, ir)
                results[m][name].append(r)
        print(f"[eval] done {name}")

    keys = ["VIF", "EN", "SSIM_vis", "SSIM_ir", "PSNR_vis", "PSNR_ir", "SF", "AG", "SCD", "Qabf", "MS_SSIM"]
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "dataset"] + keys)
        for m in methods:
            for name, _ in parsed:
                arr = np.array([np.array([r[k] for k in keys]) for r in results[m][name]])
                mean = arr.mean(0)
                std = arr.std(0)
                w.writerow([m, name] + [round(v, 4) for v in mean])
                print(m.ljust(9) + name.ljust(12) + "  ".join(f"{k}={mean[i]:.4f}" for i, k in enumerate(keys[:5])))
                print(" " * 21 + "  ".join(f"{k}={mean[i]:.4f}±{std[i]:.4f}" for i, k in enumerate(keys[5:])))
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
