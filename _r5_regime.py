"""Regime-conditional contrasts against the colour-scored visible source.

The regimes are defined from the data before any contrast is computed: the visible
source's RMS contrast (bottom and top tercile), the number of annotated pedestrians
(one against three or more), and the median annotated box area (below and above the
median). Within each subset the same estimator is used as in the main tables: pool the
per-image true-positive flags over the subset, resample the images of that subset 1000
times, and apply the same draw to both arms.
"""
import json
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "_deposit", "ssgf-llvip-fusion-detection-v1", "scripts"))
sys.path.insert(0, _here)
from recompute_from_per_image import load, pooled  # noqa: E402

PERIMG = os.environ.get("PERIMG", r"D:\CSF-main\ssgf\_restore_perimg")
STATS = os.environ.get("STATS", r"D:\CSF-main\ssgf\_restore_regime_stats.json")
DETECTORS = ["yolo11n_vis", "yolo26n_vis", "rtdetr_l_vis", "agnostic", "agn_split"]
FUSED = ["mean", "activity_only", "tophat", "probe", "tophat_permuted", "probe_permuted",
         "tophat_wshuffle", "probe_wshuffle", "ensemble", "ifcnn", "densefuse", "csf",
         "piafusion", "spgfusion"]
BOOT = 1000


def masks(stems, stats):
    rms = np.array([stats[s]["rms_contrast"] for s in stems])
    ngt = np.array([stats[s]["n_gt"] for s in stems])
    area = np.array([stats[s]["median_area"] for s in stems])
    lo, hi = np.percentile(rms, [33.3, 66.7])
    med = np.median(area[area > 0])
    out = {
        "low visible contrast": rms <= lo,
        "high visible contrast": rms >= hi,
        "single pedestrian": ngt == 1,
        "three or more pedestrians": ngt >= 3,
        "small targets": (area > 0) & (area <= med),
        "large targets": area > med,
    }
    return out


def run(det, stats):
    files = {}
    for name in ["vis"] + FUSED:
        p = os.path.join(PERIMG, f"{det}__{name}.npz")
        if os.path.exists(p):
            files[name] = load(p)
    stems = files["vis"]["stems"]
    n = len(stems)
    reg = masks(stems, stats)
    rng = np.random.default_rng(0)
    print("=" * 20, det, " (all %d images)" % n)
    for rname, m in reg.items():
        idx = np.nonzero(m)[0]
        k = len(idx)
        if k < 20:
            print("  %-26s only %d images, skipped" % (rname, k))
            continue
        draws = []
        for _ in range(BOOT):
            sel = np.zeros(n, bool)
            sel[rng.choice(idx, size=k, replace=True)] = True
            draws.append(sel)
        ref = files["vis"]
        base50 = np.array([pooled(ref, s)[0] for s in draws])
        above = below = ns = 0
        deltas = []
        for name in FUSED:
            if name not in files:
                continue
            cur = np.array([pooled(files[name], s)[0] for s in draws])
            d = cur - base50
            lo, hi = np.percentile(d, [2.5, 97.5])
            deltas.append((name, d.mean(), lo, hi))
            if lo > 0:
                above += 1
            elif hi < 0:
                below += 1
            else:
                ns += 1
        deltas.sort(key=lambda t: t[1])
        print("  %-26s n=%3d  below(resolved) %2d  above(resolved) %2d  unresolved %2d"
              % (rname, k, below, above, ns))
        for name, m_, lo, hi in deltas[:2] + deltas[-2:]:
            print("        %-18s %+.3f [%+.3f,%+.3f]" % (name, m_, lo, hi))
    print()


def main():
    stats = json.load(open(STATS))
    for det in DETECTORS:
        if os.path.exists(os.path.join(PERIMG, f"{det}__vis.npz")):
            run(det, stats)


if __name__ == "__main__":
    main()
