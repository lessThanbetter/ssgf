"""Regime split of the segmentation task, as the second-task check.

The per-image mIoU saved by `_srv_seg_regime.py` is averaged over the classes present
in the label of that image, which is the standard convention. The main text's
segmentation table was produced with the stricter convention of its own script, which
counts a class absent from both prediction and label as an IoU of zero, so its absolute
values are lower by construction; the comparison here is within one convention.

The visible source's RMS contrast splits the 361 MSRS-test images into terciles, and
each fused input's paired difference against the unfused visible source is computed
within each tercile with image-resampled intervals.
"""
import json
import os
import sys

import numpy as np

OUT = r"D:\CSF-main\ssgf\_restore_seg_regime_summary.json"
DATA = r"D:\CSF-main\ssgf\_restore_seg_regime.json"
INPUTS = ["infrared source", "mean", "activity_only", "tophat", "probe",
          "tophat_permuted", "probe_permuted", "tophat_wshuffle", "probe_wshuffle"]
BOOT = 2000


def main():
    d = json.load(open(DATA))
    stems = sorted(d)
    c = np.array([d[s]["rms_contrast"] for s in stems])
    lo, hi = np.percentile(c, [33.3, 66.7])
    groups = {
        "low visible contrast (bottom tercile)": c <= lo,
        "middle tercile": (c > lo) & (c < hi),
        "high visible contrast (top tercile)": c >= hi,
        "all 361 images": np.ones(len(stems), bool),
    }
    rng = np.random.default_rng(0)
    out = {"thresholds": [float(lo), float(hi)], "groups": {}}
    print("mIoU convention: mean over the classes present in each label")
    print("tercile thresholds on visible RMS contrast: %.3f and %.3f" % (lo, hi))
    for gname, m in groups.items():
        idx = np.nonzero(m)[0]
        k = len(idx)
        draws = [rng.choice(idx, size=k, replace=True) for _ in range(BOOT)]
        base = np.array([np.mean([d[stems[i]]["miou"]["visible source"] for i in dr])
                         for dr in draws])
        res = {}
        print("=" * 16, gname, "n=%d" % k)
        for name in INPUTS:
            cur = np.array([np.mean([d[stems[i]]["miou"][name] for i in dr])
                            for dr in draws])
            diff = cur - base
            l, h = np.percentile(diff, [2.5, 97.5])
            star = "*" if (l > 0 or h < 0) else " "
            res[name] = {"delta": round(float(diff.mean()), 4),
                         "lo": round(float(l), 4), "hi": round(float(h), 4),
                         "miou": round(float(np.mean([d[stems[i]]["miou"][name] for i in idx])), 4)}
            print("   %-18s mIoU %.4f  vs visible source %+.4f [%+.4f,%+.4f]%s"
                  % (name, res[name]["miou"], res[name]["delta"], l, h, star))
        res["visible source"] = {"delta": 0.0, "lo": 0.0, "hi": 0.0,
                                 "miou": round(float(np.mean([d[stems[i]]["miou"]["visible source"]
                                                              for i in idx])), 4)}
        out["groups"][gname] = {"n": int(k), "inputs": res}
    json.dump(out, open(OUT, "w"), indent=2)
    print("SEG_REGIME_SUMMARY_DONE", flush=True)


if __name__ == "__main__":
    main()
