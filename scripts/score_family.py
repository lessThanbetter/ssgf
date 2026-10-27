"""Score every fused input under every visible-trained checkpoint, with intervals.

The estimator is the one the released analysis and the main tables use, so the rows
it reproduces for the first architecture are a self-check: per image the predictions
are matched to the ground truth at the ten IoU thresholds with the same Hungarian
assignment as `analysis/j1_eval2.py`, the per-image statistics are pooled over the
300 images and the average precision comes from the library's own `ap_per_class`.

For every (checkpoint, input) the script writes
  _restore/perimg/<ckpt>__<input>.npz   the per-image statistics, for the deposit
  _restore/scores.json                  pooled mAP50 and mAP50-95, and the paired
                                        bootstrap contrast against the visible source
Existing pairs are skipped, so the job can be restarted.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/ssgf")

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.metrics import ap_per_class, box_iou  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

BASE = "/root/autodl-tmp"
TEST = f"{BASE}/datasets/det/test"
EF = f"{BASE}/datasets/det/eval_fused"
K1 = f"{BASE}/ssgf/data/k1_native/imgs"
SO = f"{BASE}/ssgf/sota_outs"
OUT = f"{BASE}/ssgf/_restore"
PERIMG = f"{OUT}/perimg"
IOU_T = np.linspace(0.5, 0.95, 10)
BOOT = 1000

INPUTS = [
    ("vis", f"{TEST}/visible"),
    ("ir", f"{TEST}/infrared"),
    ("mean", f"{K1}/mean"),
    ("activity_only", f"{K1}/activity_only"),
    ("tophat", f"{K1}/tophat"),
    ("probe", f"{K1}/probe"),
    ("tophat_permuted", f"{K1}/tophat_permuted"),
    ("probe_permuted", f"{K1}/probe_permuted"),
    ("tophat_wshuffle", f"{K1}/tophat_wshuffle"),
    ("probe_wshuffle", f"{K1}/probe_wshuffle"),
    ("ensemble", f"{EF}/ensemble/images"),
    ("ifcnn", f"{EF}/ifcnn/images"),
    ("densefuse", f"{EF}/densefuse/images"),
    ("csf", f"{SO}/csf_llvip_native"),
    ("piafusion", f"{SO}/pia_llvip_native_fp16"),
    ("spgfusion", f"{SO}/spg_llvip_native_fp16"),
]

CKPTS = [
    ("yolo11n_vis", f"{BASE}/ssgf/data/det_runs/llvip/weights/best.pt"),
    ("agnostic", f"{BASE}/ssgf/runs/detect/data/det_agnostic/agn/weights/best.pt"),
    ("vis_split", f"{BASE}/ssgf/_split_runs/vis_split/weights/best.pt"),
    ("agn_split", f"{BASE}/ssgf/_split_runs/agn_split/weights/best.pt"),
    ("yolo26n_vis", f"{BASE}/ssgf/data/det_runs/llvip_y26/weights/best.pt"),
    ("rtdetr_l_vis", f"{BASE}/ssgf/data/det_runs/llvip_rtdetr/weights/best.pt"),
]


def gt_boxes(stem):
    boxes, classes = [], []
    p = os.path.join(TEST, "labels", stem + ".txt")
    if not os.path.exists(p):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    for line in open(p):
        q = line.split()
        if len(q) < 5:
            continue
        c, x, y, w, h = int(float(q[0])), *map(float, q[1:5])
        boxes.append([(x - w / 2) * 1280, (y - h / 2) * 1024,
                      (x + w / 2) * 1280, (y + h / 2) * 1024])
        classes.append(c)
    return (np.asarray(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32),
            np.asarray(classes, np.int64) if classes else np.zeros((0,), np.int64))


def match(pred_boxes, pred_cls, gt_b, gt_c):
    n = len(pred_boxes)
    correct = np.zeros((n, len(IOU_T)), dtype=bool)
    if n == 0 or len(gt_b) == 0:
        return correct
    iou = box_iou(torch.from_numpy(pred_boxes), torch.from_numpy(gt_b)).numpy()
    correct_cls = pred_cls[:, None] == gt_c[None, :]
    for j, t in enumerate(IOU_T):
        x = np.logical_and(iou >= t, correct_cls)
        if x.any():
            i, g = linear_sum_assignment(np.logical_not(x))
            correct[i, j] = x[i, g]
    return correct


def pooled_metric(pool, idx=None):
    stats = pool if idx is None else [pool[i] for i in idx]
    stats = [s for s in stats if s is not None]
    if not stats:
        return 0.0, 0.0
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


def find(d, stem):
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def run_ckpt(name, path, stems, gts, summary):
    model = YOLO(path)
    n = len(stems)
    pool = {}
    for iname, d in INPUTS:
        npz = os.path.join(PERIMG, f"{name}__{iname}.npz")
        done = summary.get(name, {}).get(iname)
        if done and done.get("missing", 1) == 0 and os.path.exists(npz):
            print(f"  {name} {iname}: already complete, skip", flush=True)
            continue
        rows = []
        missing = 0
        t0 = time.time()
        for stem in stems:
            p = find(d, stem)
            if p is None:
                rows.append(None)
                missing += 1
                continue
            try:
                r = model.predict(p, conf=0.001, imgsz=640, device=0,
                                  verbose=False)[0]
            except Exception as exc:  # a file still being written, or unreadable
                print("    read failure on %s (%s), treated as absent"
                      % (p, type(exc).__name__), flush=True)
                rows.append(None)
                missing += 1
                continue
            pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            pc = r.boxes.cls.cpu().numpy().astype(np.int64)
            cf = r.boxes.conf.cpu().numpy().astype(np.float32)
            gt_b, gt_c = gts[stem]
            rows.append((match(pb, pc, gt_b, gt_c), cf, pc, gt_c))
        if missing:
            print("  %-12s %-16s SKIPPED, %d of %d images absent"
                  % (name, iname, missing, n), flush=True)
            continue
        pool[iname] = rows
        m50, m95 = pooled_metric(rows)
        np.savez_compressed(
            npz,
            tp=np.concatenate([r[0] for r in rows if r is not None], axis=0)
            if any(r is not None for r in rows) else np.zeros((0, 10), bool),
            conf=np.concatenate([r[1] for r in rows if r is not None], axis=0)
            if any(r is not None for r in rows) else np.zeros((0,), np.float32),
            pcls=np.concatenate([r[2] for r in rows if r is not None], axis=0)
            if any(r is not None for r in rows) else np.zeros((0,), np.int64),
            img_idx=np.concatenate([np.full(len(r[1]), i, np.int32)
                                    for i, r in enumerate(rows) if r is not None])
            if any(r is not None for r in rows) else np.zeros((0,), np.int32),
            tcls=np.concatenate([r[3] for r in rows if r is not None], axis=0)
            if any(r is not None for r in rows) else np.zeros((0,), np.int64),
            stems=np.array(stems), n_img=n)
        summary.setdefault(name, {})[iname] = {
            "mAP50": round(m50, 4), "mAP50_95": round(m95, 4), "missing": missing}
        print("  %-12s %-16s mAP50 %.4f  mAP50-95 %.4f  missing %d  (%.0fs)"
              % (name, iname, m50, m95, missing, time.time() - t0), flush=True)

    if "vis" not in pool:
        print(f"[{name}] visible source missing, no intervals computed", flush=True)
        json.dump(summary, open(f"{OUT}/scores.json", "w"), indent=2)
        return

    # paired bootstrap of every input against the visible source
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, n, size=n) for _ in range(BOOT)]
    base50 = np.array([pooled_metric(pool["vis"], d)[0] for d in draws])
    base95 = np.array([pooled_metric(pool["vis"], d)[1] for d in draws])
    for iname, _d in INPUTS:
        if iname == "vis" or iname not in pool:
            continue
        c50 = np.array([pooled_metric(pool[iname], d)[0] for d in draws])
        c95 = np.array([pooled_metric(pool[iname], d)[1] for d in draws])
        d50, d95 = c50 - base50, c95 - base95
        summary[name][iname].update({
            "delta50": round(float(d50.mean()), 4),
            "lo50": round(float(np.percentile(d50, 2.5)), 4),
            "hi50": round(float(np.percentile(d50, 97.5)), 4),
            "delta95": round(float(d95.mean()), 4),
            "lo95": round(float(np.percentile(d95, 2.5)), 4),
            "hi95": round(float(np.percentile(d95, 97.5)), 4)})
    json.dump(summary, open(f"{OUT}/scores.json", "w"), indent=2)


def main():
    os.makedirs(PERIMG, exist_ok=True)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{TEST}/labels"))
    gts = {s: gt_boxes(s) for s in stems}
    summary = {}
    if os.path.exists(f"{OUT}/scores.json"):
        summary = json.load(open(f"{OUT}/scores.json"))
    for name, path in CKPTS:
        if not os.path.exists(path):
            print(f"[skip] {name}: no checkpoint at {path}", flush=True)
            continue
        complete = [k for k, v in summary.get(name, {}).items() if v.get("missing", 1) == 0]
        if len(complete) == len(INPUTS):
            print(f"[skip] {name}: already scored", flush=True)
            continue
        print(f"=== {name} ===", flush=True)
        run_ckpt(name, path, stems, gts, summary)
    print("SCORE_FAMILY_DONE", flush=True)


if __name__ == "__main__":
    main()
