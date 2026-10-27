"""Regime-conditional contrasts, with slice-based pooling for speed.

Regimes are defined from the data before any contrast is computed: the visible
source's RMS contrast (bottom and top tercile of the 300 images), the number of
annotated pedestrians (one, against three or more), and the median annotated box area
(below and above the median). Within each subset the estimator of the main tables is
used, the images of the subset are resampled 400 times, and the same draw is applied
to both arms.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics.utils.metrics import ap_per_class  # noqa: E402

PERIMG = os.environ.get("PERIMG", "/root/autodl-tmp/ssgf/_restore/perimg")
STATS = os.environ.get("STATS", "/root/autodl-tmp/ssgf/_restore/regime_stats.json")
OUT = os.environ.get("OUT", "/root/autodl-tmp/ssgf/_restore/regime_conditional.json")
DETECTORS = ["yolo11n_vis", "yolo26n_vis", "rtdetr_l_vis", "agnostic", "vis_split", "agn_split"]
FUSED = ["mean", "activity_only", "tophat", "probe", "tophat_permuted", "probe_permuted",
         "tophat_wshuffle", "probe_wshuffle", "ensemble", "ifcnn", "densefuse", "csf",
         "piafusion", "spgfusion"]
BOOT = 400


def load(path):
    z = np.load(path)
    n = int(z["n_img"])
    img = z["img_idx"]
    order = np.argsort(img, kind="stable")
    tp, conf, pcls = z["tp"][order], z["conf"][order], z["pcls"][order]
    bounds = np.searchsorted(img[order], np.arange(n + 1))
    n_gt = np.asarray(z["n_gt"])
    cuts = np.concatenate(([0], np.cumsum(n_gt)))
    tcls = z["tcls"]
    per_tp = [tp[bounds[i]:bounds[i + 1]] for i in range(n)]
    per_cf = [conf[bounds[i]:bounds[i + 1]] for i in range(n)]
    per_cl = [pcls[bounds[i]:bounds[i + 1]] for i in range(n)]
    per_gt = [tcls[cuts[i]:cuts[i + 1]] for i in range(n)]
    return {"tp": per_tp, "cf": per_cf, "cl": per_cl, "gt": per_gt, "n": n,
            "stems": list(z["stems"])}


def pooled(st, idx):
    if len(idx) == 0:
        return 0.0, 0.0
    tp = np.concatenate([st["tp"][i] for i in idx])
    cf = np.concatenate([st["cf"][i] for i in idx])
    cl = np.concatenate([st["cl"][i] for i in idx])
    gt = np.concatenate([st["gt"][i] for i in idx])
    if len(tp) == 0 or len(gt) == 0:
        return 0.0, 0.0
    out = ap_per_class(tp, cf, cl, gt, names={0: "person"})
    ap = np.asarray(out[5])
    if ap.size == 0:
        return 0.0, 0.0
    return float(ap[:, 0].mean()), float(ap.mean())


def regimes(stems, stats):
    rms = np.array([stats[s]["rms_contrast"] for s in stems])
    ngt = np.array([stats[s]["n_gt"] for s in stems])
    area = np.array([stats[s]["median_area"] for s in stems])
    lo, hi = np.percentile(rms, [33.3, 66.7])
    med = float(np.median(area[area > 0]))
    return {
        "low visible contrast (bottom tercile)": rms <= lo,
        "high visible contrast (top tercile)": rms >= hi,
        "single pedestrian": ngt == 1,
        "three or more pedestrians": ngt >= 3,
        "small targets (below median area)": (area > 0) & (area <= med),
        "large targets (above median area)": area > med,
    }


def main():
    stats = json.load(open(STATS))
    rng = np.random.default_rng(0)
    out = {}
    for det in DETECTORS:
        p = os.path.join(PERIMG, f"{det}__vis.npz")
        if not os.path.exists(p):
            print(f"[skip] {det}", flush=True)
            continue
        ref = load(p)
        inputs = {name: load(os.path.join(PERIMG, f"{det}__{name}.npz"))
                  for name in FUSED if os.path.exists(os.path.join(PERIMG, f"{det}__{name}.npz"))}
        stems = ref["stems"]
        n = ref["n"]
        reg = regimes(stems, stats)
        out[det] = {}
        print("=" * 18, det, flush=True)
        for rname, m in reg.items():
            idx = np.nonzero(m)[0]
            k = len(idx)
            if k < 20:
                print("  %-38s only %d images, skipped" % (rname, k), flush=True)
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
            print("  %-38s n=%3d  below(resolved) %2d  above(resolved) %2d  unresolved %2d"
                  % (rname, k, below, above, len(res) - below - above), flush=True)
            for name in ("spgfusion", "mean", "probe", "activity_only"):
                v = res.get(name)
                if v:
                    print("        %-14s %+.3f [%+.3f,%+.3f]" % (name, v["delta"], v["lo"], v["hi"]),
                          flush=True)
            json.dump(out, open(OUT, "w"), indent=2)
    print("REGIME_DONE", flush=True)


if __name__ == "__main__":
    main()
