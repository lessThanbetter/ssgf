"""Unified LLVIP detection evaluation using Ultralytics' own AP machinery.

Design. For every input and every detector we predict per image, match the
predictions to the ground truth with the same greedy IoU matching Ultralytics
uses in validation, and then compute the metric with `ap_per_class`, i.e. the
same function that produces the Ultralytics mAP reported in the paper. This
removes the estimator mismatch between the table values and the text deltas.

Paired bootstrap intervals resample images and recompute the metric from the
pooled per-image statistics, so the interval belongs to the same estimator as
the point estimate.

Usage (from /root/autodl-tmp/ssgf):
    python j1_eval2.py --detector vis --conf 0.001
    python j1_eval2.py --detector agn --conf 0.001
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/root/autodl-tmp/ssgf")
from ultralytics.utils.metrics import ap_per_class, box_iou  # noqa: E402

BASE = "/root/autodl-tmp"
TEST = f"{BASE}/datasets/det/test"
OUTDIR = f"{BASE}/ssgf/data/j1_eval"

INPUTS = [
    ("vis", f"{TEST}/visible"),
    ("ir", f"{TEST}/infrared"),
    ("mean", f"{BASE}/ssgf/data/k1_native/imgs/mean"),
    ("activity_only", f"{BASE}/ssgf/data/k1_native/imgs/activity_only"),
    ("tophat", f"{BASE}/ssgf/data/k1_native/imgs/tophat"),
    ("probe", f"{BASE}/ssgf/data/k1_native/imgs/probe"),
    ("tophat_permuted", f"{BASE}/ssgf/data/k1_native/imgs/tophat_permuted"),
    ("probe_permuted", f"{BASE}/ssgf/data/k1_native/imgs/probe_permuted"),
    ("tophat_wshuffle", f"{BASE}/ssgf/data/k1_native/imgs/tophat_wshuffle"),
    ("probe_wshuffle", f"{BASE}/ssgf/data/k1_native/imgs/probe_wshuffle"),
    ("ensemble", f"{BASE}/ssgf/sota_outs/ensemble_llvip_native"),
    ("ifcnn", f"{BASE}/ssgf/sota_outs/ifcnn_llvip_native"),
    ("densefuse", f"{BASE}/ssgf/sota_outs/densefuse_llvip_native"),
    ("piafusion", f"{BASE}/ssgf/sota_outs/pia_llvip_native"),
    ("spgfusion", f"{BASE}/ssgf/sota_outs/spg_llvip_native"),
    ("csf", f"{BASE}/ssgf/sota_outs/csf_llvip_native"),
]

DETECTORS = {
    "vis": f"{BASE}/ssgf/data/det_runs/llvip/weights/best.pt",
    "agn": f"{BASE}/ssgf/data/det_agnostic/agn/weights/best.pt",
}

IOU_T = np.linspace(0.5, 0.95, 10)


def gt_boxes(stem):
    boxes, classes = [], []
    p = os.path.join(TEST, "labels", stem + ".txt")
    if not os.path.exists(p):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    with open(p) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            boxes.append([(x - w / 2) * 1280, (y - h / 2) * 1024,
                          (x + w / 2) * 1280, (y + h / 2) * 1024])
            classes.append(c)
    return (np.asarray(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32),
            np.asarray(classes, np.int64) if classes else np.zeros((0,), np.int64))


def find_image(d, stem):
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def match(pred_boxes, pred_cls, gt_b, gt_c):
    """Return bool [n_pred, 10] TP matrix, as in Ultralytics validation."""
    n = len(pred_boxes)
    correct = np.zeros((n, len(IOU_T)), dtype=bool)
    if n == 0 or len(gt_b) == 0:
        return correct
    iou = box_iou(torch.from_numpy(pred_boxes),
                  torch.from_numpy(gt_b)).numpy()
    correct_cls = pred_cls[:, None] == gt_c[None, :]
    for j, t in enumerate(IOU_T):
        x = np.logical_and(iou >= t, correct_cls)
        if x.any():
            i, g = linear_sum_assignment(np.logical_not(x))
            correct[i, j] = x[i, g]
    return correct


def pooled_metric(pool, idx=None):
    stats = pool if idx is None else [pool[i] for i in idx]
    tp = np.concatenate([s[0] for s in stats], axis=0)
    conf = np.concatenate([s[1] for s in stats], axis=0)
    pcls = np.concatenate([s[2] for s in stats], axis=0)
    tcls = np.concatenate([s[3] for s in stats], axis=0)
    if len(tp) == 0 or len(tcls) == 0:
        return 0.0, 0.0
    out = ap_per_class(tp, conf, pcls, tcls, names={0: "person"})
    ap = np.asarray(out[5])
    if ap.size == 0:
        return 0.0, 0.0
    return float(ap[:, 0].mean()), float(ap.mean())


def run(det_name, conf_thr, boot, limit=0):
    from ultralytics import YOLO
    model = YOLO(DETECTORS[det_name])
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{TEST}/labels"))
    if limit:
        stems = stems[:limit]
    n = len(stems)
    print(f"[{det_name}] {n} images, conf={conf_thr}", flush=True)

    gts = {s: gt_boxes(s) for s in stems}
    pool = {name: [] for name, _ in INPUTS}
    usable = {name: [] for name, _ in INPUTS}   # indices with GT present
    for k, stem in enumerate(stems):
        gt_b, gt_c = gts[stem]
        for name, d in INPUTS:
            p = find_image(d, stem)
            if p is None:
                pool[name].append(None)
                continue
            img = np.asarray(Image.open(p).convert("L").resize((1280, 1024)), np.uint8)
            bgr = np.stack([img] * 3, axis=-1)
            r = model.predict(bgr, conf=conf_thr, imgsz=640, device=0, verbose=False)[0]
            pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            pc = r.boxes.cls.cpu().numpy().astype(np.int64)
            cf = r.boxes.conf.cpu().numpy().astype(np.float32)
            pool[name].append((match(pb, pc, gt_b, gt_c), cf, pc, gt_c))
        if (k + 1) % 50 == 0:
            print(f"  [{det_name}] {k+1}/{n}", flush=True)

    scores = {}
    for name, _ in INPUTS:
        stats = [s for s in pool[name] if s is not None]
        m50, m95 = pooled_metric(stats)
        scores[name] = {"mAP50": m50, "mAP50_95": m95}
        print(f"  {name:16s} mAP50={m50:.4f} mAP50-95={m95:.4f}", flush=True)

    rng = np.random.RandomState(0)
    draws = [rng.randint(0, n, n) for _ in range(boot)]
    cache = {}
    for name, _ in INPUTS:
        stats_all = pool[name]
        keep = [i for i, s in enumerate(stats_all) if s is not None]
        stats = [stats_all[i] for i in keep]
        remap = {orig: pos for pos, orig in enumerate(keep)}
        v = np.empty(boot)
        for b, d in enumerate(draws):
            v[b] = pooled_metric(stats, [remap[j] for j in d if j in remap])[0]
        cache[name] = v

    pairs = [("probe", "mean"), ("tophat", "mean"), ("probe", "activity_only"),
             ("tophat", "activity_only"), ("ensemble", "mean"),
             ("mean", "vis"), ("probe", "vis"), ("tophat", "vis"),
             ("ifcnn", "vis"), ("densefuse", "vis"), ("piafusion", "vis"),
             ("spgfusion", "vis"),
             ("csf", "vis"), ("csf", "mean"),
             ("probe", "probe_permuted"), ("tophat", "tophat_permuted"),
             ("probe", "probe_wshuffle"), ("tophat", "tophat_wshuffle")]
    deltas = {}
    print(f"  -- paired bootstrap, {boot} resamples --", flush=True)
    for a, b in pairs:
        d = cache[a] - cache[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        star = "*" if (lo > 0 or hi < 0) else " "
        deltas[f"{a}-{b}"] = {"delta": float(d.mean()), "lo": float(lo),
                              "hi": float(hi), "significant": bool(lo > 0 or hi < 0)}
        print(f"  {a:16s} - {b:16s} d={d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {star}",
              flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(f"{OUTDIR}/unified_det_{det_name}.json", "w") as fh:
        json.dump({"config": {"conf": conf_thr, "imgsz": 640, "boot": boot,
                              "n_images": n,
                              "detector": DETECTORS[det_name]},
                   "scores": scores, "deltas": deltas}, fh, indent=2)
    print(f"[{det_name}] saved unified_det_{det_name}.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="vis", choices=["vis", "agn"])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.detector, a.conf, a.boot, a.limit)
    print("J1_EVAL2_DONE")


if __name__ == "__main__":
    main()
