"""J3: detector seed replication.

Trains the visible-trained YOLO11n three more times with seeds 1, 2 and 3 on the
same 1500-image subset and the same recipe as the seed-0 run, then scores a fixed
subset of inputs with every seed. This prices detector training variance against
the differences reported in Table tab:det.

Usage (from /root/autodl-tmp/ssgf):
    python j3_det_seeds.py --task train
    python j3_det_seeds.py --task eval
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/ssgf")
from j1_eval2 import TEST, find_image, gt_boxes, match, pooled_metric  # noqa: E402

BASE = "/root/autodl-tmp"
SEEDS = [1, 2, 3]
OUTDIR = f"{BASE}/ssgf/data/j3_seeds"
INPUTS = [
    ("vis", f"{TEST}/visible"),
    ("ir", f"{TEST}/infrared"),
    ("mean", f"{BASE}/ssgf/data/k1_native/imgs/mean"),
    ("activity_only", f"{BASE}/ssgf/data/k1_native/imgs/activity_only"),
    ("tophat", f"{BASE}/ssgf/data/k1_native/imgs/tophat"),
    ("probe", f"{BASE}/ssgf/data/k1_native/imgs/probe"),
    ("piafusion", f"{BASE}/ssgf/sota_outs/pia_llvip_native"),
    ("spgfusion", f"{BASE}/ssgf/sota_outs/spg_llvip_native"),
]


def train():
    from ultralytics import YOLO
    for s in SEEDS:
        out = f"{BASE}/ssgf/data/det_runs/llvip_seed{s}"
        if os.path.exists(f"{out}/weights/best.pt"):
            print(f"seed {s}: exists, skip", flush=True)
            continue
        print(f"=== training seed {s} ===", flush=True)
        m = YOLO(f"{BASE}/ultralytics-main/yolo11n.pt")
        m.train(data=f"{BASE}/datasets/det/yolo.yaml", epochs=40, imgsz=640,
                batch=16, seed=s, device=0, workers=4, verbose=False,
                project=f"{BASE}/ssgf/data/det_runs", name=f"llvip_seed{s}",
                exist_ok=True)
    print("J3_TRAIN_DONE", flush=True)


def evaluate(boot=500):
    from ultralytics import YOLO
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{TEST}/labels"))
    gts = {s: gt_boxes(s) for s in stems}
    models = {"seed0": f"{BASE}/ssgf/data/det_runs/llvip/weights/best.pt"}
    for s in SEEDS:
        p = f"{BASE}/ssgf/data/det_runs/llvip_seed{s}/weights/best.pt"
        if os.path.exists(p):
            models[f"seed{s}"] = p
    res = {}
    for mname, path in models.items():
        model = YOLO(path)
        pool = {name: [] for name, _ in INPUTS}
        for k, stem in enumerate(stems):
            gt_b, gt_c = gts[stem]
            for name, d in INPUTS:
                p = find_image(d, stem)
                if p is None:
                    pool[name].append(None)
                    continue
                img = np.asarray(Image.open(p).convert("L").resize((1280, 1024)),
                                 np.uint8)
                bgr = np.stack([img] * 3, axis=-1)
                r = model.predict(bgr, conf=0.001, imgsz=640, device=0,
                                  verbose=False)[0]
                pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                pc = r.boxes.cls.cpu().numpy().astype(np.int64)
                cf = r.boxes.conf.cpu().numpy().astype(np.float32)
                pool[name].append((match(pb, pc, gt_b, gt_c), cf, pc, gt_c))
            if (k + 1) % 100 == 0:
                print(f"  [{mname}] {k+1}/{len(stems)}", flush=True)
        res[mname] = {}
        for name, _ in INPUTS:
            stats = [s for s in pool[name] if s is not None]
            m50, m95 = pooled_metric(stats)
            res[mname][name] = {"mAP50": m50, "mAP50_95": m95}
            print(f"  {mname} {name:16s} mAP50={m50:.4f}", flush=True)
    # seed-to-seed spread of the spgfusion - vis contrast
    deltas = [res[m]["spgfusion"]["mAP50"] - res[m]["vis"]["mAP50"] for m in res]
    spread = {"spgfusion_minus_vis": deltas,
              "mean": float(np.mean(deltas)), "std": float(np.std(deltas)),
              "min": float(np.min(deltas)), "max": float(np.max(deltas))}
    print(f"  spgfusion-vis across seeds: {spread}", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(f"{OUTDIR}/seed_replication.json", "w") as fh:
        json.dump({"scores": res, "contrast": spread}, fh, indent=2)
    print("J3_EVAL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="train", choices=["train", "eval"])
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    train() if a.task == "train" else evaluate(a.boot)
