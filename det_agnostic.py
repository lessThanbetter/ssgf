"""Modality-agnostic detector: train on VIS+IR, eval mAP on sources and fused.

Runs out the concern that the probe's downstream detection gain is only a
visible-modality bias. A detector trained on both modalities is evaluated on VIS,
IR, and probe-fused test images under the same LLVIP protocol.

Usage:
    python det_agnostic.py --det /root/autol-tmp/datasets/det --ae data/msrs_ae.pt \
        --probe data/msrs_probe.pt --out data/det_agnostic
"""

from __future__ import annotations

import argparse, os, shutil
import numpy as np
from PIL import Image
import torch

from ssgf.config import get_default_config
from ssgf.models import Encoder, Decoder, SemanticProbe
from ssgf.fusion import SemanticSaliencyRule
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--ae", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", default="data/det_agnostic")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) build a combined VIS+IR training set with prefixed names
    train = os.path.join(args.det, "train")
    comb = os.path.join(args.out, "train_agn")
    os.makedirs(os.path.join(comb, "images"), exist_ok=True)
    os.makedirs(os.path.join(comb, "labels"), exist_ok=True)
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(train, "visible")))
    for n in names:
        for mod, tag in (("visible", "vis"), ("infrared", "ir")):
            src_img = os.path.join(train, mod, n + ".jpg")
            src_lab = os.path.join(train, "labels", n + ".txt")
            shutil.copy(src_img, os.path.join(comb, "images", f"{tag}_{n}.jpg"))
            shutil.copy(src_lab, os.path.join(comb, "labels", f"{tag}_{n}.txt"))
    print(f"[agn] combined train images: {len(os.listdir(os.path.join(comb,'images')))}")

    yaml = f"path: {comb}\ntrain: images\nval: images\nnc: 1\nnames: {{0: person}}\n"
    yp = os.path.join(args.out, "agn.yaml")
    open(yp, "w").write(yaml)

    from ultralytics import YOLO
    det = YOLO("/root/autodl-tmp/ultralytics-main/yolo11n.pt")
    det.train(data=yp, epochs=args.epochs, imgsz=640, batch=16, device=0, project=args.out, name="agn", workers=4, verbose=False)
    model = YOLO(os.path.join(args.out, "agn", "weights", "best.pt"))

    # 2) eval on sources + probe-fused
    cfg = get_default_config(); enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae, encoder=enc, decoder=dec)
    rule = SemanticSaliencyRule(enc, dec, cfg, dev); load_ckpt(args.probe, probe=rule.probe)
    test = os.path.join(args.det, "test")
    labels = sorted(f for f in os.listdir(os.path.join(test, "labels")))
    base = os.path.join(args.out, "eval")
    for source in ["vis", "ir", "fused"]:
        img_dir = os.path.join(base, source, "images"); lab_dir = os.path.join(base, source, "labels")
        os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
        for lab in labels:
            name = lab.replace(".txt", ".jpg")
            vis = imread(os.path.join(test, "visible", name)); ir = imread(os.path.join(test, "infrared", name))
            if source == "vis":
                out_img = vis
            elif source == "ir":
                out_img = ir
            else:
                v = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0); i = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0)
                f, _ = rule.fuse_pair(v, i); out_img = np.clip(f.squeeze().cpu().numpy(), 0, 1)
            Image.fromarray((np.clip(out_img, 0, 1) * 255).astype("uint8")).save(os.path.join(img_dir, name))
            shutil.copy(os.path.join(test, "labels", lab), os.path.join(lab_dir, lab))
        ey = f"path: {os.path.join(base, source)}\ntrain: images\nval: images\nnc: 1\nnames: {{0: person}}\n"
        eyp = os.path.join(base, source, "data.yaml"); open(eyp, "w").write(ey)
        res = model.val(data=eyp, imgsz=640, batch=16, device=0, verbose=False, project=base, name=f"{source}_val")
        print(f"[agn] {source:6s} mAP50={getattr(res.box,'map50',0):.4f}  mAP50-95={getattr(res.box,'map',0):.4f}")


if __name__ == "__main__":
    main()
