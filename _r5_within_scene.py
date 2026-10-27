"""Within-scene check of the regime effect.

Visible-source contrast is a property of the scene, so the contrast bins of
`_r5_regime_curve.py` also separate the LLVIP sequences. This checks the effect inside
each sequence: every scene with at least 30 test images is split at its own median
contrast, and the paired contrast against the colour-scored visible source is computed
for a few rules in each half with image-resampled intervals.
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_ap import load, pooled  # noqa: E402

PERIMG = r"D:\CSF-main\ssgf\_restore_perimg"
STATS = r"D:\CSF-main\ssgf\_restore_regime_stats.json"
OUT = r"D:\CSF-main\ssgf\_restore_within_scene.json"
RULES = ["spgfusion", "ifcnn", "mean", "probe", "activity_only", "tophat"]
BOOT = 400


def analyse(det, stats, rng):
    ref = load(os.path.join(PERIMG, f"{det}__vis.npz"))
    inputs = {n: load(os.path.join(PERIMG, f"{det}__{n}.npz")) for n in RULES}
    stems = ref["stems"]
    rms = {s: stats[s]["rms_contrast"] for s in stems}
    scenes = collections.defaultdict(list)
    for i, s in enumerate(stems):
        scenes[s[:3]].append(i)
    out = {}
    print("=" * 18, det)
    for sc, idx in sorted(scenes.items(), key=lambda kv: -len(kv[1])):
        if len(idx) < 30:
            continue
        c = np.array([rms[stems[i]] for i in idx])
        med = float(np.median(c))
        lo = [i for i in idx if rms[stems[i]] <= med]
        hi = [i for i in idx if rms[stems[i]] > med]
        row = {"n": len(idx), "n_low": len(lo), "n_high": len(hi),
               "contrast_median": round(med, 4), "low": {}, "high": {}}
        for half, ids in (("low", lo), ("high", hi)):
            k = len(ids)
            draws = [rng.choice(ids, size=k, replace=True) for _ in range(BOOT)]
            base = np.array([pooled(ref, d)[0] for d in draws])
            for name, st in inputs.items():
                cur = np.array([pooled(st, d)[0] for d in draws])
                ds = cur - base
                row[half][name] = {"delta": round(float(ds.mean()), 4),
                                   "lo": round(float(np.percentile(ds, 2.5)), 4),
                                   "hi": round(float(np.percentile(ds, 97.5)), 4)}
        out[sc] = row
        print("  scene %s  n=%d (low %d / high %d)  contrast median %.3f"
              % (sc, len(idx), len(lo), len(hi), med))
        for name in RULES:
            a, b = row["low"][name], row["high"][name]
            print("      %-13s low %+.3f [%+.3f,%+.3f]   high %+.3f [%+.3f,%+.3f]   low-high %+.3f"
                  % (name, a["delta"], a["lo"], a["hi"], b["delta"], b["lo"], b["hi"],
                     a["delta"] - b["delta"]))
    return out


def main():
    stats = json.load(open(STATS))
    rng = np.random.default_rng(0)
    out = {}
    for det in ("yolo11n_vis", "agnostic"):
        out[det] = analyse(det, stats, rng)
    json.dump(out, open(OUT, "w"), indent=2)
    print("WITHIN_SCENE_DONE", flush=True)


if __name__ == "__main__":
    main()
