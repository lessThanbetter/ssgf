"""Downstream detection for IFCNN / DenseFuse baselines (same LLVIP protocol)."""

from __future__ import annotations

import argparse, os, shutil
import numpy as np
from PIL import Image
import torch

from ssgf.models.baselines import IFCNN, DenseFuse, fuse_l1
from ssgf.utils.io import imread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--btl", required=True, help="dir with ifcnn.pt / densefuse.pt")
    ap.add_argument("--model", required=True, help="YOLO best.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {"ifcnn": IFCNN(), "densefuse": DenseFuse()}
    for k, m in models.items():
        m.to(dev).eval()
        m.load_state_dict(torch.load(os.path.join(args.btl, f"{k}.pt"), map_location="cpu"))

    test = os.path.join(args.det, "test")
    labels = sorted(os.listdir(os.path.join(test, "labels")))
    base = os.path.join(args.det, "eval_fused")
    print(f"[det-baseline] {len(labels)} test images")
    for k, m in models.items():
        img_dir = os.path.join(base, k, "images"); lab_dir = os.path.join(base, k, "labels")
        os.makedirs(img_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
        for lab in labels:
            name = lab.replace(".txt", ".jpg")
            vis = imread(os.path.join(test, "visible", name))
            ir = imread(os.path.join(test, "infrared", name))
            v = (torch.from_numpy(vis) * 2 - 1).unsqueeze(0).unsqueeze(0).to(dev)
            i = (torch.from_numpy(ir) * 2 - 1).unsqueeze(0).unsqueeze(0).to(dev)
            f = m.decode(fuse_l1(m.encode(v), m.encode(i))).detach().squeeze().cpu().numpy()
            f = np.clip((f + 1) / 2, 0, 1)
            Image.fromarray((f * 255).astype("uint8")).save(os.path.join(img_dir, name))
            shutil.copy(os.path.join(test, "labels", lab), os.path.join(lab_dir, lab))
        yaml = f"path: {os.path.join(base, k)}\ntrain: images\nval: images\nnc: 1\nnames: {{0: person}}\n"
        yp = os.path.join(base, k, "data.yaml")
        open(yp, "w").write(yaml)
        from ultralytics import YOLO
        det = YOLO(args.model)
        res = det.val(data=yp, imgsz=640, batch=16, device=0, verbose=False, project=base, name=f"{k}_val")
        print(f"{k:10s} mAP50={getattr(res.box,'map50',0):.4f}  mAP50-95={getattr(res.box,'map',0):.4f}")


if __name__ == "__main__":
    main()
