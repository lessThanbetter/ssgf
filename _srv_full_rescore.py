"""One unified pass over every row of Supplementary Tables S12 and S13.

Same 300 LLVIP test images, four checkpoints (the two main detectors and the two
retrained on the clean split) and the eleven inputs of the two audit tables, scored
with the same matcher, the same 101-point pooled AP and the same paired bootstrap
against the colour visible source that `paired_intervals_audit.py` and
`native_rescore.py` used, so every number in the two tables and in the interval
paragraph comes from a single run.

Writes /root/autodl-tmp/ssgf/_audit/unified_rescore.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/ssgf")

from ultralytics import YOLO  # noqa: E402

ROOT = "/root/autodl-tmp/ssgf"
TEST = "/root/autodl-tmp/datasets/det/test"
EF = "/root/autodl-tmp/datasets/det/eval_fused"
H, W = 1024, 1280
BOOT = 1000

CKPTS = [
    ("visTrained", f"{ROOT}/data/det_runs/llvip/weights/best.pt"),
    ("agnostic", f"{ROOT}/runs/detect/data/det_agnostic/agn/weights/best.pt"),
    ("vis_split", f"{ROOT}/_split_runs/vis_split/weights/best.pt"),
    ("agn_split", f"{ROOT}/_split_runs/agn_split/weights/best.pt"),
]
INPUTS = [
    ("vis_colour", f"{TEST}/visible"),
    ("vis_gray", f"{ROOT}/_audit/vis_gray/images"),
    ("ir", f"{TEST}/infrared"),
    ("mean", f"{EF}/mean/images"),
    ("tophat", f"{EF}/tophat/images"),
    ("probe", f"{EF}/probe/images"),
    ("ensemble", f"{EF}/ensemble/images"),
    ("ifcnn", f"{EF}/ifcnn/images"),
    ("densefuse", f"{EF}/densefuse/images"),
    ("piafusion", f"{ROOT}/sota_outs/pia_llvip_native_fp16"),
    ("spgfusion", f"{ROOT}/sota_outs/spg_llvip_native_fp16"),
]


def gt_for(stem):
    p = f"{TEST}/labels/{stem}.txt"
    b = []
    if os.path.exists(p):
        for line in open(p):
            q = line.split()
            if len(q) < 5:
                continue
            x, y, w, h = map(float, q[1:5])
            b.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
    return np.asarray(b, np.float32) if b else np.zeros((0, 4), np.float32)


def iou_matrix(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), np.float32)
    p, g = pred[:, None, :], gt[None, :, :]
    x1 = np.maximum(p[..., 0], g[..., 0])
    y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2])
    y2 = np.minimum(p[..., 3], g[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    return inter / (ap + ag - inter + 1e-9)


def find(d, stem):
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def ap50(conf, tp, n_gt):
    if n_gt <= 0 or len(conf) == 0:
        return 0.0
    conf, tp = np.asarray(conf, np.float32), np.asarray(tp, np.float32)
    order = np.argsort(-conf)
    tpc = np.cumsum(tp[order])
    fpc = np.cumsum(1 - tp[order])
    rec = tpc / n_gt
    prec = tpc / (tpc + fpc + 1e-9)
    xs = np.linspace(0, 1, 101)
    return float(np.mean([prec[rec >= x].max() if (rec >= x).any() else 0.0 for x in xs]))


def pooled(per_img, idx):
    conf, tp, ngt = [], [], 0
    for i in idx:
        conf.extend(per_img[i][0])
        tp.extend(per_img[i][1])
        ngt += per_img[i][2]
    return ap50(conf, tp, ngt)


def main():
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{TEST}/visible"))
    gts = {s: gt_for(s) for s in stems}
    idx_all = list(range(len(stems)))
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(stems), size=len(stems)) for _ in range(BOOT)]
    out = {}
    for cname, ck in CKPTS:
        if not os.path.exists(ck):
            print("missing", cname, flush=True)
            continue
        model = YOLO(ck)
        per = {}
        for name, d in INPUTS:
            missing = 0
            rows = []
            for s in stems:
                p = find(d, s)
                if p is None:
                    rows.append(([], [], 0))
                    missing += 1
                    continue
                r = model.predict(p, conf=0.001, imgsz=640, device=0, verbose=False)[0]
                pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                cf = r.boxes.conf.cpu().numpy().astype(np.float32)
                o = np.argsort(-cf)
                pb, cf = pb[o], cf[o]
                gt = gts[s]
                flags = np.zeros(len(pb), np.float32)
                used = np.zeros(len(gt), bool)
                if len(pb) and len(gt):
                    M = iou_matrix(pb, gt)
                    for i in range(len(pb)):
                        mask = np.where(used, -1.0, M[i])
                        j = int(np.argmax(mask))
                        if mask[j] >= 0.5:
                            flags[i] = 1.0
                            used[j] = True
                rows.append((cf.tolist(), flags.tolist(), len(gt)))
            per[name] = rows
            print("  %-10s %-11s AP50 %.4f  missing %d"
                  % (cname, name, pooled(rows, idx_all), missing), flush=True)

        base_ap = pooled(per["vis_colour"], idx_all)
        base_boot = np.array([pooled(per["vis_colour"], d) for d in draws])
        entry = {}
        for name, _d in INPUTS:
            if name == "vis_colour":
                continue
            ap = pooled(per[name], idx_all)
            cur = np.array([pooled(per[name], d) for d in draws])
            diff = cur - base_boot
            lo, hi = np.percentile(diff, 2.5), np.percentile(diff, 97.5)
            entry[name] = {"ap50": round(ap, 4),
                           "delta": round(float(diff.mean()), 4),
                           "lo": round(float(lo), 4),
                           "hi": round(float(hi), 4)}
            print("  %-10s %-11s %.4f delta %+.4f [%+.4f,%+.4f]"
                  % (cname, name, ap, diff.mean(), lo, hi), flush=True)
        out[cname] = {"vis_colour_ap50": round(base_ap, 4), "inputs": entry}
        json.dump(out, open(f"{ROOT}/_audit/unified_rescore.json", "w"), indent=2)
        print("wrote unified_rescore.json after", cname, flush=True)
    print("UNIFIED_RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
