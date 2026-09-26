"""Fuse with the evaluator-robust models and score them under every evaluator.

Usage (from /root/autodl-tmp/ssgf):
    python robust_eval.py --task fuse
    python robust_eval.py --task eval
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/ssgf")
from j1_eval2 import match, pooled_metric            # noqa: E402
from robust_eval_common import (BASE, FLIR, MODELS, OUT, RES,  # noqa: E402
                                build, fuse_pair)


def stems_llvip():
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(f"{BASE}/datasets/det/test/labels"))


def stems_flir():
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(f"{FLIR}/test/labels"))


def fuse():
    from ssgf.utils.io import imread
    for name, ckpt in MODELS.items():
        if not os.path.exists(ckpt):
            print(f"{name}: no checkpoint, skip", flush=True)
            continue
        cfg, enc, dec, probe = build(ckpt)
        # LLVIP
        ss = stems_llvip()
        d = f"{OUT}/{name}_llvip_native"
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        for k, s in enumerate(ss):
            v = imread(f"{BASE}/datasets/det/test/visible/{s}.jpg")
            r = imread(f"{BASE}/datasets/det/test/infrared/{s}.jpg")
            Image.fromarray((fuse_pair(cfg, enc, dec, probe, v, r) * 255)
                            .astype(np.uint8)).save(f"{d}/{s}.png")
            if (k + 1) % 100 == 0:
                print(f"  {name} llvip {k+1}/{len(ss)} {time.time()-t0:.0f}s",
                      flush=True)
        # FLIR
        ss = stems_flir()
        d = f"{OUT}/flir_{name}"
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        for k, s in enumerate(ss):
            v = imread(f"{FLIR}/test/visible/{s}.jpg")
            r = imread(f"{FLIR}/test/infrared/{s}.jpg")
            Image.fromarray((fuse_pair(cfg, enc, dec, probe, v, r) * 255)
                            .astype(np.uint8)).save(f"{d}/{s}.png")
            if (k + 1) % 100 == 0:
                print(f"  {name} flir {k+1}/{len(ss)} {time.time()-t0:.0f}s",
                      flush=True)
        print(f"{name}: fused", flush=True)
    print("ROBUST_FUSE_DONE", flush=True)


LLVIP_EVAL = [
    ("yolo11n", f"{BASE}/ssgf/data/det_runs/llvip/weights/best.pt"),
    ("yolo26n", f"{BASE}/ssgf/data/det_runs/llvip_y26/weights/best.pt"),
    ("rtdetr", f"{BASE}/ssgf/data/det_runs/llvip_rtdetr/weights/best.pt"),
    ("agnostic", f"{BASE}/ssgf/data/det_agnostic/agn/weights/best.pt"),
]
FLIR_EVAL = [
    ("yolo11n", f"{BASE}/ssgf/data/det_runs/flir_vis/weights/best.pt"),
    ("yolo26n", f"{BASE}/ssgf/data/det_runs/flir_y26/weights/best.pt"),
    ("agnostic", f"{BASE}/ssgf/data/det_runs/flir_agn/weights/best.pt"),
]


def score(model_path, inputs, test_root, hw, tag):
    from ultralytics import YOLO
    model = YOLO(model_path)
    H, W = hw
    stems = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(f"{test_root}/labels"))

    def gt(stem):
        b, c = [], []
        p = f"{test_root}/labels/{stem}.txt"
        if os.path.exists(p):
            for line in open(p):
                q = line.split()
                if len(q) < 5:
                    continue
                x, y, w, h = map(float, q[1:5])
                b.append([(x - w / 2) * W, (y - h / 2) * H,
                          (x + w / 2) * W, (y + h / 2) * H])
                c.append(0)
        return (np.asarray(b, np.float32) if b else np.zeros((0, 4), np.float32),
                np.asarray(c, np.int64) if c else np.zeros((0,), np.int64))

    gts = {s: gt(s) for s in stems}
    out = {}
    for name, d in inputs:
        stats = []
        for stem in stems:
            p = None
            for ext in (".png", ".jpg"):
                q = os.path.join(d, stem + ext)
                if os.path.exists(q):
                    p = q
                    break
            if p is None:
                continue
            img = np.asarray(Image.open(p).convert("L").resize((W, H)), np.uint8)
            bgr = np.stack([img] * 3, axis=-1)
            r = model.predict(bgr, conf=0.001, imgsz=640, device=0, verbose=False)[0]
            pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            pc = r.boxes.cls.cpu().numpy().astype(np.int64)
            cf = r.boxes.conf.cpu().numpy().astype(np.float32)
            stats.append((match(pb, pc, *gts[stem]), cf, pc, gts[stem][1]))
        m50, _ = pooled_metric(stats)
        out[name] = m50
        print(f"  {tag} {name:16s} mAP50={m50:.4f}", flush=True)
    return out


def eval_():
    llvip_inputs = [
        ("vis", f"{BASE}/datasets/det/test/visible"),
        ("ir", f"{BASE}/datasets/det/test/infrared"),
        ("mean", f"{BASE}/ssgf/data/k1_native/imgs/mean"),
        ("probe", f"{BASE}/ssgf/data/k1_native/imgs/probe"),
        ("e2ecsf", f"{OUT}/e2ecsf_llvip_native"),
        ("dettgt", f"{OUT}/dettgtar_llvip_native"),
        ("robrr", f"{OUT}/robrr_llvip_native"),
        ("robwc", f"{OUT}/robwc_llvip_native"),
        ("proberr", f"{OUT}/proberr_llvip_native"),
        ("probeone", f"{OUT}/probeone_llvip_native"),
    ]
    flir_inputs = [
        ("vis", f"{FLIR}/test/visible"),
        ("ir", f"{FLIR}/test/infrared"),
        ("mean", f"{OUT}/flir_mean"),
        ("probe", f"{OUT}/flir_probe"),
        ("robrr", f"{OUT}/flir_robrr"),
        ("robwc", f"{OUT}/flir_robwc"),
        ("proberr", f"{OUT}/flir_proberr"),
        ("probeone", f"{OUT}/flir_probeone"),
    ]
    res = {"llvip": {}, "flir": {}}
    for ename, epath in LLVIP_EVAL:
        if not os.path.exists(epath):
            continue
        print(f"== LLVIP evaluator {ename} ==", flush=True)
        res["llvip"][ename] = score(epath, llvip_inputs,
                                    f"{BASE}/datasets/det/test", (1024, 1280),
                                    ename)
    for ename, epath in FLIR_EVAL:
        if not os.path.exists(epath):
            continue
        print(f"== FLIR evaluator {ename} ==", flush=True)
        res["flir"][ename] = score(epath, flir_inputs, f"{FLIR}/test",
                                   (512, 640), ename)

    summ = {}
    for bench, evals in res.items():
        names = sorted({m for e in evals.values() for m in e})
        print(f"\n== {bench}: mean and worst case across evaluators ==", flush=True)
        for m in names:
            vals = [evals[e][m] for e in evals if m in evals[e]]
            if not vals:
                continue
            summ[f"{bench}/{m}"] = {"mean": float(np.mean(vals)),
                                    "min": float(np.min(vals)),
                                    "per_evaluator": {e: evals[e][m]
                                                      for e in evals
                                                      if m in evals[e]}}
            print(f"  {m:16s} mean={np.mean(vals):.4f} min={np.min(vals):.4f}",
                  flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/robust.json", "w") as fh:
        json.dump({"raw": res, "summary": summ}, fh, indent=2)
    print("ROBUST_EVAL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["fuse", "eval"])
    a = ap.parse_args()
    fuse() if a.task == "fuse" else eval_()
