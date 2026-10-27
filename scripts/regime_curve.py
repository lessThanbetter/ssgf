"""Continuous regime curve: the deficit as a function of visible-source contrast.

Bins the 300 LLVIP test images by the RMS contrast of the unfused visible source, six
bins of fifty images, and computes for each bin the paired contrast of every fused
input against the colour-scored visible source, resampling the images of the bin 400
times and applying the same draw to both arms. Two detectors are used: the primary
visible-trained YOLO11n and the modality-agnostic detector, so the deficit and the
reversal can be read off the same axis.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_ap import load, pooled  # noqa: E402

PERIMG = r"D:\CSF-main\ssgf\_restore_perimg"
STATS = r"D:\CSF-main\ssgf\_restore_regime_stats.json"
OUT = r"D:\CSF-main\ssgf\_restore_regime_curve.json"
FUSED = ["mean", "activity_only", "tophat", "probe", "tophat_permuted", "probe_permuted",
         "tophat_wshuffle", "probe_wshuffle", "ensemble", "ifcnn", "densefuse", "csf",
         "piafusion", "spgfusion"]
DETECTORS = ["yolo11n_vis", "agnostic"]
BINS = 6
BOOT = 400


def main():
    stats = json.load(open(STATS))
    rng = np.random.default_rng(0)
    out = {}
    for det in DETECTORS:
        ref = load(os.path.join(PERIMG, f"{det}__vis.npz"))
        inputs = {n: load(os.path.join(PERIMG, f"{det}__{n}.npz"))
                  for n in FUSED if os.path.exists(os.path.join(PERIMG, f"{det}__{n}.npz"))}
        stems = ref["stems"]
        rms = np.array([stats[s]["rms_contrast"] for s in stems])
        order = np.argsort(rms)
        edges = np.array_split(order, BINS)
        out[det] = []
        print("=" * 18, det)
        for b, idx in enumerate(edges):
            k = len(idx)
            draws = [rng.choice(idx, size=k, replace=True) for _ in range(BOOT)]
            base = np.array([pooled(ref, d)[0] for d in draws])
            res = {}
            for name, st in inputs.items():
                cur = np.array([pooled(st, d)[0] for d in draws])
                ds = cur - base
                cd = np.array([pooled(st, d)[0] for d in draws])  # placeholder
                res[name] = {"delta": round(float(ds.mean()), 4),
                             "lo": round(float(np.percentile(ds, 2.5)), 4),
                             "hi": round(float(np.percentile(ds, 97.5)), 4)}
            below = sum(1 for v in res.values() if v["hi"] < 0)
            above = sum(1 for v in res.values() if v["lo"] > 0)
            out[det].append({"bin": b + 1, "n": k,
                             "contrast_lo": round(float(rms[idx].min()), 4),
                             "contrast_hi": round(float(rms[idx].max()), 4),
                             "mean_contrast": round(float(rms[idx].mean()), 4),
                             "below": below, "above": above,
                             "unresolved": len(res) - below - above, "inputs": res})
            print("  bin %d  contrast %.3f-%.3f  n=%d  below %2d  above %2d  unres %2d"
                  % (b + 1, rms[idx].min(), rms[idx].max(), k, below, above,
                     len(res) - below - above))
            for name in ("spgfusion", "ifcnn", "mean", "probe", "activity_only"):
                v = res.get(name)
                if v:
                    star = "*" if (v["lo"] > 0 or v["hi"] < 0) else " "
                    print("        %-13s %+.3f [%+.3f,%+.3f]%s"
                          % (name, v["delta"], v["lo"], v["hi"], star))
            json.dump(out, open(OUT, "w"), indent=2)
    print("REGIME_CURVE_DONE", flush=True)


if __name__ == "__main__":
    main()
