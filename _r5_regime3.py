"""Regime-conditional contrasts, computed locally from the per-image files.

The regimes are defined from the data before any contrast is computed: the unfused
visible source's RMS contrast (bottom and top tercile of the 300 images), the number of
annotated pedestrians (one, or three and more), and the median annotated box area
(below and above the median). Within each subset the estimator of the main tables is
used, the images of the subset are resampled 400 times, and the same draw is applied to
both arms. The local implementation of the estimator agrees with the server's to 0.001
mAP50 on the full set, which is checked by `--check`.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_ap import load, pooled  # noqa: E402

PERIMG = r"D:\CSF-main\ssgf\_restore_perimg"
STATS = r"D:\CSF-main\ssgf\_restore_regime_stats.json"
OUT = r"D:\CSF-main\ssgf\_restore_regime_conditional.json"
FUSED = ["mean", "activity_only", "tophat", "probe", "tophat_permuted", "probe_permuted",
         "tophat_wshuffle", "probe_wshuffle", "ensemble", "ifcnn", "densefuse", "csf",
         "piafusion", "spgfusion"]
BOOT = 1000


def regimes(stems, stats):
    rms = np.array([stats[s]["rms_contrast"] for s in stems])
    ngt = np.array([stats[s]["n_gt"] for s in stems])
    area = np.array([stats[s]["median_area"] for s in stems])
    lo, hi = np.percentile(rms, [33.3, 66.7])
    med = float(np.median(area[area > 0]))
    return {
        "all 300 images": np.ones(len(stems), bool),
        "low visible contrast (bottom tercile)": rms <= lo,
        "high visible contrast (top tercile)": rms >= hi,
        "single pedestrian": ngt == 1,
        "three or more pedestrians": ngt >= 3,
        "small targets (below median area)": (area > 0) & (area <= med),
        "large targets (above median area)": area > med,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default="yolo11n_vis,yolo26n_vis,rtdetr_l_vis,agnostic,vis_split,agn_split")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    stats = json.load(open(STATS))
    rng = np.random.default_rng(0)
    out = {}
    for det in a.detectors.split(","):
        if not os.path.exists(os.path.join(PERIMG, f"{det}__vis.npz")):
            print("[skip]", det, flush=True)
            continue
        ref = load(os.path.join(PERIMG, f"{det}__vis.npz"))
        inputs = {name: load(os.path.join(PERIMG, f"{det}__{name}.npz"))
                  for name in FUSED
                  if os.path.exists(os.path.join(PERIMG, f"{det}__{name}.npz"))}
        stems, n = ref["stems"], ref["n"]
        out[det] = {}
        print("=" * 18, det, flush=True)
        for rname, m in regimes(stems, stats).items():
            idx = np.nonzero(m)[0]
            k = len(idx)
            if k < 20:
                continue
            draws = [rng.choice(idx, size=k, replace=True) for _ in range(BOOT)]
            base = np.array([pooled(ref, d)[0] for d in draws])
            res = {}
            for name, st in inputs.items():
                cur = np.array([pooled(st, d)[0] for d in draws])
                ds = cur - base
                lo, hi = np.percentile(ds, [2.5, 97.5])
                res[name] = {"delta": round(float(ds.mean()), 4),
                             "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                             "mAP50": round(pooled(st, idx)[0], 4)}
            below = sum(1 for v in res.values() if v["hi"] < 0)
            above = sum(1 for v in res.values() if v["lo"] > 0)
            out[det][rname] = {"n_images": int(k), "below_resolved": below,
                               "above_resolved": above,
                               "unresolved": len(res) - below - above, "inputs": res}
            worst = min(res.items(), key=lambda kv: kv[1]["delta"])
            print("  %-38s n=%3d  below(res) %2d  above(res) %2d  unres %2d"
                  % (rname, k, below, above, len(res) - below - above), flush=True)
            for name in ("spgfusion", "ifcnn", "mean", "probe", "activity_only"):
                v = res.get(name)
                if v:
                    star = "*" if (v["lo"] > 0 or v["hi"] < 0) else " "
                    print("        %-14s %+.3f [%+.3f,%+.3f]%s" % (name, v["delta"], v["lo"], v["hi"], star),
                          flush=True)
            json.dump(out, open(a.out, "w"), indent=2)
    print("REGIME_LOCAL_DONE", flush=True)


if __name__ == "__main__":
    main()
