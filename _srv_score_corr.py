"""Corruption robustness: score every input after degrading it in memory.

Three degradations are applied to every input, including the unfused sources, so the
contrast stays paired: additive Gaussian noise (sigma 0.05), Gaussian blur (sigma 1.5
px) and JPEG compression at quality 30. Nothing is written to disk except the per-image
statistics, and the primary visible-trained detector and the modality-agnostic detector
are both used.
"""
import io
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, "/root/autodl-tmp/ssgf")

from _restore_score import INPUTS, gt_boxes, match  # noqa: E402
from ultralytics import YOLO  # noqa: E402

BASE = "/root/autodl-tmp"
TEST = f"{BASE}/datasets/det/test"
OUT = f"{BASE}/ssgf/_restore/perimg"
CKPTS = [("yolo11n_vis", f"{BASE}/ssgf/data/det_runs/llvip/weights/best.pt"),
         ("agnostic", f"{BASE}/ssgf/data/det_runs/agnostic/weights/best.pt")]
CONDS = ["noise", "blur", "jpeg30"]
RNG = np.random.default_rng(0)


def find(d, stem):
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def degrade(path, cond):
    im = Image.open(path).convert("L")
    if cond == "noise":
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = np.clip(a + RNG.normal(0, 0.05, a.shape), 0, 1)
        return Image.fromarray((a * 255).astype("uint8"))
    if cond == "blur":
        return im.filter(ImageFilter.GaussianBlur(1.5))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=30)
    buf.seek(0)
    return Image.open(buf).convert("L")


def main():
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{TEST}/labels"))
    gts = {s: gt_boxes(s) for s in stems}
    n = len(stems)
    n_gt = np.array([sum(1 for _ in open(f"{TEST}/labels/{s}.txt")) for s in stems], np.int32)
    for ck_name, ck in CKPTS:
        if not os.path.exists(ck):
            print(f"[skip] {ck_name}: no checkpoint", flush=True)
            continue
        model = YOLO(ck)
        for cond in CONDS:
            name = f"{ck_name}_{cond}"
            print(f"=== {name} ===", flush=True)
            for iname, d in INPUTS:
                npz = os.path.join(OUT, f"{name}__{iname}.npz")
                if os.path.exists(npz):
                    continue
                rows = []
                for stem in stems:
                    p = find(d, stem)
                    if p is None:
                        rows.append(None)
                        continue
                    r = model.predict(degrade(p, cond), conf=0.001, imgsz=640,
                                      device=0, verbose=False)[0]
                    pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                    pc = r.boxes.cls.cpu().numpy().astype(np.int64)
                    cf = r.boxes.conf.cpu().numpy().astype(np.float32)
                    gt_b, gt_c = gts[stem]
                    rows.append((match(pb, pc, gt_b, gt_c), cf, pc, gt_c))
                keep = [r for r in rows if r is not None]
                if len(keep) != n:
                    print(f"  {iname}: {n - len(keep)} absent, skipped", flush=True)
                    continue
                np.savez_compressed(
                    npz,
                    tp=np.concatenate([r[0] for r in keep], axis=0),
                    conf=np.concatenate([r[1] for r in keep], axis=0),
                    pcls=np.concatenate([r[2] for r in keep], axis=0),
                    img_idx=np.concatenate([np.full(len(r[1]), i, np.int32)
                                            for i, r in enumerate(rows) if r is not None]),
                    tcls=np.concatenate([r[3] for r in keep], axis=0),
                    n_gt=n_gt, stems=np.array(stems), n_img=n)
                print(f"  {iname}: written", flush=True)
    print("SCORE_CORR_DONE", flush=True)


if __name__ == "__main__":
    main()
