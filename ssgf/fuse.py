"""Inference entry point.

Example (CPU smoke test on the repo's own test images):
    python -m ssgf.fuse --root D:/CSF-main --rule semantic --init random \
        --out D:/CSF-main/ssgf/outputs --metrics

On the server with trained weights:
    python -m ssgf.fuse --root /data/tno --rule semantic \
        --ckpt /data/ssgf/ae.pt --out /data/ssgf/out --metrics
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .config import get_default_config
from .fusion import MeanRule, ModalityCSRule, SemanticPriorRule, SemanticSaliencyRule
from .models import Decoder, Encoder, build_prior
from .utils.checkpoint import load_ckpt
from .utils.io import find_pairs, imread, imsave
from .utils.metrics import report


def _parse_cfg(pairs: list[str]) -> dict:
    out = {}
    for kv in pairs:
        k, _, v = kv.partition("=")
        out[k.strip()] = _cast(v.strip())
    return out


def _cast(v: str):
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _crop_to_min(a: np.ndarray, b: np.ndarray):
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def main() -> None:
    ap = argparse.ArgumentParser(description="SSGF fusion inference")
    ap.add_argument("--root", required=True)
    ap.add_argument("--vis_dir", default="vis")
    ap.add_argument("--ir_dir", default="ir")
    ap.add_argument("--rule", default="semantic", choices=["semantic", "modality_cs", "mean", "semantic_prior"])
    ap.add_argument("--ckpt", default=None, help="encoder/decoder (+probe) checkpoint")
    ap.add_argument("--prior", default=None, help="U2-Net ONNX path for --rule semantic_prior")
    ap.add_argument("--init", default="trained", choices=["trained", "random"], help="fallback if no ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--save_maps", action="store_true", help="save saliency weight maps")
    ap.add_argument("--limit", type=int, default=0, help="only fuse first N pairs (0 = all)")
    ap.add_argument("--cfg", nargs="*", default=[], help="key=value config override")
    args = ap.parse_args()

    overrides = _parse_cfg(args.cfg)
    cfg = get_default_config(**overrides)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    encoder, decoder = Encoder(cfg), Decoder(cfg)
    if args.ckpt:
        payload = load_ckpt(args.ckpt, encoder=encoder, decoder=decoder)
        print(f"[load] encoder/decoder from {args.ckpt}")
    else:
        print(f"[init] random weights ({args.init}) -- use --ckpt for trained results")

    if args.rule == "mean":
        rule = MeanRule(encoder, decoder, cfg, device)
    elif args.rule == "modality_cs":
        rule = ModalityCSRule(encoder, decoder, cfg, device)
    elif args.rule == "semantic_prior":
        if not args.prior:
            raise SystemExit("--rule semantic_prior requires --prior <u2net.onnx>")
        prior = build_prior(args.prior)
        rule = SemanticPriorRule(encoder, decoder, cfg, prior, device)
    else:
        prior = build_prior(args.prior) if args.prior else None
        rule = SemanticSaliencyRule(encoder, decoder, cfg, device, prior=prior)
        if args.ckpt:
            load_ckpt(args.ckpt, probe=rule.probe)

    pairs = find_pairs(args.root, args.vis_dir, args.ir_dir)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"[data] {len(pairs)} pairs from {args.root}/{args.vis_dir} + {args.ir_dir}")

    results = []
    for vis_path, ir_path in pairs:
        name = os.path.splitext(os.path.basename(vis_path))[0]
        vis, ir = imread(vis_path), imread(ir_path)
        vis, ir = _crop_to_min(vis, ir)
        vt = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        it = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0)
        fused, debug = rule.fuse_pair(vt, it)

        f = fused.squeeze().cpu().numpy()
        imsave(os.path.join(args.out, f"{name}.png"), f)
        if args.save_maps:
            wv = debug["w_vis"].mean(dim=1).squeeze().cpu().numpy()  # (H,W) averaged over channels
            wi = debug["w_ir"].mean(dim=1).squeeze().cpu().numpy()
            imsave(os.path.join(args.out, f"w_vis_{name}.png"), wv)
            imsave(os.path.join(args.out, f"w_ir_{name}.png"), wi)
        if args.metrics:
            results.append(report(f, vis, ir))

    if args.metrics:
        keys = results[0].keys()
        print("\n=== mean metrics over %d pairs ===" % len(results))
        for k in keys:
            vals = [r[k] for r in results]
            print(f"{k:10s}: {np.mean(vals):.4f}  +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
