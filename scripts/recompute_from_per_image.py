"""Rebuild any detection table and its paired intervals from the per-image files.

Usage
    python recompute_from_per_image.py --dir ../data/per_image --detector yolo11n_vis
    python recompute_from_per_image.py --dir ../data/per_image --detector yolo11n_vis --ref vis

The estimator is the one the manuscript's detection tables use: the per-image
true-positive flags come from a one-to-one maximum-cardinality assignment to the
ground truth of the same image at each of the ten IoU thresholds, the flags are
pooled over the resampled set of images, and the average precision is computed with
the detector library's own function.
"""
import argparse
import glob
import json
import os

import numpy as np

try:
    from ultralytics.utils.metrics import ap_per_class
except ImportError:  # allow the script to state the requirement instead of failing obscurely
    ap_per_class = None


def load(path):
    """One file per detector and input, with the per-image layout documented above."""
    z = np.load(path)
    n = int(z["n_img"])
    tp, conf, pcls, img = z["tp"], z["conf"], z["pcls"], z["img_idx"]
    order = np.argsort(img, kind="stable")
    tp, conf, pcls, img = tp[order], conf[order], pcls[order], img[order]
    n_gt = np.asarray(z["n_gt"]) if "n_gt" in z else None
    tcls = z["tcls"]
    if n_gt is None:   # older files: split the ground truth evenly is not possible
        raise SystemExit("this file lacks the per-image ground-truth counts (n_gt)")
    cuts = np.concatenate(([0], np.cumsum(n_gt)))
    return {"tp": tp, "conf": conf, "pcls": pcls, "img": img, "n_gt": n_gt,
            "cuts": cuts, "tcls": tcls, "stems": list(z["stems"]), "n": n}


def pooled(stats, sel):
    """Pool the detections of the selected images and return mAP50 and mAP50-95."""
    mask = sel[stats["img"]]
    tp, conf, pcls = stats["tp"][mask], stats["conf"][mask], stats["pcls"][mask]
    cuts = stats["cuts"]
    idx = np.nonzero(sel)[0]
    gt = np.concatenate([stats["tcls"][cuts[i]:cuts[i + 1]] for i in idx])
    if len(tp) == 0 or len(gt) == 0:
        return 0.0, 0.0
    out = ap_per_class(tp, conf, pcls, gt, names={0: "person"})
    ap = np.asarray(out[5])
    if ap.size == 0:
        return 0.0, 0.0
    return float(ap[:, 0].mean()), float(ap.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--ref", default="vis")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if ap_per_class is None:
        raise SystemExit("this script needs `ultralytics` installed to compute mAP")

    files = sorted(glob.glob(os.path.join(a.dir, f"{a.detector}__*.npz")))
    if not files:
        raise SystemExit(f"no per-image files for {a.detector} in {a.dir}")
    stats, names = {}, []
    for f in files:
        name = os.path.basename(f)[len(a.detector) + 2:-4]
        stats[name] = load(f)
        names.append(name)
    if a.ref not in stats:
        raise SystemExit(f"reference input {a.ref} not present")
    n = stats[a.ref]["n"]
    allsel = np.ones(n, bool)
    rng = np.random.default_rng(0)
    masks = []
    for _ in range(a.boot):
        sel = np.zeros(n, bool)
        sel[rng.integers(0, n, size=n)] = True
        masks.append(sel)
    ref = stats[a.ref]
    base50 = np.array([pooled(ref, s)[0] for s in masks])
    base95 = np.array([pooled(ref, s)[1] for s in masks])

    rows = {}
    print("%-16s %8s %8s   %-24s %-24s" % ("input", "mAP50", "mAP95", "d50 [lo,hi]", "d95 [lo,hi]"))
    for name in sorted(names):
        s = stats[name]
        m50, m95 = pooled(s, allsel)
        if name == a.ref:
            print("%-16s %8.4f %8.4f   reference" % (name, m50, m95))
            rows[name] = {"mAP50": round(m50, 4), "mAP50_95": round(m95, 4)}
            continue
        c50 = np.array([pooled(s, sel)[0] for sel in masks]) - base50
        c95 = np.array([pooled(s, sel)[1] for sel in masks]) - base95
        lo50, hi50 = np.percentile(c50, [2.5, 97.5])
        lo95, hi95 = np.percentile(c95, [2.5, 97.5])
        star = "*" if (lo50 > 0 or hi50 < 0) else " "
        print("%-16s %8.4f %8.4f   %+.4f [%+.4f,%+.4f]%s   %+.4f [%+.4f,%+.4f]"
              % (name, m50, m95, c50.mean(), lo50, hi50, star, c95.mean(), lo95, hi95))
        rows[name] = {"mAP50": round(m50, 4), "mAP50_95": round(m95, 4),
                      "delta50": round(float(c50.mean()), 4),
                      "lo50": round(float(lo50), 4), "hi50": round(float(hi50), 4),
                      "delta95": round(float(c95.mean()), 4),
                      "lo95": round(float(lo95), 4), "hi95": round(float(hi95), 4)}
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
