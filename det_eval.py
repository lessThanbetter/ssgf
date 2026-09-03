"""Detection mAP downstream evaluation on fused images (LLVIP subset).

For each input type (visible source / infrared source / mean / tophat / probe /
ensemble fused) it produces a temporary val set and runs the trained detector,
reporting mAP50 and mAP50-95.

Usage (server):
    python det_eval.py --det /root/autodl-tmp/datasets/det \
        --model data/det_runs/llvip/weights/best.pt \
        --ae data/msrs_ae.pt --probe data/msrs_probe.pt
"""

from __future__ import annotations

import argparse, os, shutil
import numpy as np
from PIL import Image
import torch

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, SemanticProbe, build_prior
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="path to det (contains test/{visible,infrared,labels})")
    ap.add_argument("--model", required=True, help="trained YOLO best.pt")
    ap.add_argument("--ae", required=True)
    ap.add_argument("--probe", required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae, encoder=enc, decoder=dec)
    prior = build_prior("tophat")
    mean_rule = MeanRule(enc, dec, cfg, dev)
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt(args.probe, probe=probe_rule.probe)

    test = os.path.join(args.det, "test")
    labels = sorted(os.listdir(os.path.join(test, "labels")))
    print(f"[det-eval] {len(labels)} test images")
    base_dir = os.path.join(args.det, "eval_fused")

    methods = ["vis", "ir", "mean", "tophat", "probe", "ensemble"]
    out = {}
    for method in methods:
        img_dir = os.path.join(base_dir, method, "images")
        lab_dir = os.path.join(base_dir, method, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        for lab in labels:
            name = lab.replace(".txt", ".jpg")
            vis = imread(os.path.join(test, "visible", name))
            ir = imread(os.path.join(test, "infrared", name))
            if method == "vis":
                f_img = vis
            elif method == "ir":
                f_img = ir
            else:
                v = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0)
                i = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0)
                if method == "mean":
                    f, _ = mean_rule.fuse_pair(v, i)
                elif method == "tophat":
                    f, _ = top_rule.fuse_pair(v, i)
                elif method == "probe":
                    f, _ = probe_rule.fuse_pair(v, i)
                else:
                    f1, _ = top_rule.fuse_pair(v, i)
                    f2, _ = probe_rule.fuse_pair(v, i)
                    f = (f1 + f2) / 2.0
                f_img = f.squeeze().cpu().numpy()
            Image.fromarray((np.clip(f_img, 0, 1) * 255).astype("uint8")).save(os.path.join(img_dir, name))
            shutil.copy(os.path.join(test, "labels", lab), os.path.join(lab_dir, lab))
        yaml_path = os.path.join(base_dir, method, "data.yaml")
        with open(yaml_path, "w") as fh:
            fh.write(f"path: {os.path.join(base_dir, method)}\ntrain: images\nval: images\nnc: 1\nnames: {{0: person}}\n")
        from ultralytics import YOLO
        det = YOLO(args.model)
        m = det.val(data=yaml_path, imgsz=640, batch=16, device=0, verbose=False, project=base_dir, name=f"{method}_val")
        mp = getattr(m.box, "map50", 0.0)
        m_ap = getattr(m.box, "map", 0.0)
        out[method] = (mp, m_ap)
        print(f"{method.ljust(9)} mAP50={mp:.4f}  mAP50-95={m_ap:.4f}")
    print("\n=== summary ===")
    for k, (mp, m_ap) in out.items():
        print(f"{k.ljust(9)} mAP50={mp:.4f}  mAP50-95={m_ap:.4f}")


if __name__ == "__main__":
    main()
