"""Contrasts for the new detectors: YOLO11s and the two inference resolutions.

Same estimator as the paper (pooled AP over the paired image draws). Writes
`_restore_yolo11s.json` and `_restore_resolution.json`.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_ap import load, pooled  # noqa: E402

PERIMG = r"D:\CSF-main\ssgf\_restore_perimg"
FUSED = ["mean", "activity_only", "tophat", "probe", "tophat_permuted", "probe_permuted",
         "tophat_wshuffle", "probe_wshuffle", "ensemble", "ifcnn", "densefuse", "csf",
         "piafusion", "spgfusion"]


def contrasts(tag, draws, seed=0, extra=("ir",)):
    ref = load(os.path.join(PERIMG, "%s__vis.npz" % tag))
    n = ref["n"]
    rng = np.random.default_rng(seed)
    idx = [rng.choice(n, size=n, replace=True) for _ in range(draws)]
    base50 = np.array([pooled(ref, d)[0] for d in idx])
    base95 = np.array([pooled(ref, d)[1] for d in idx])
    out = {"tag": tag, "n_images": n, "draws": draws,
           "vis": {"mAP50": float(pooled(ref, np.arange(n))[0]),
                   "mAP50_95": float(pooled(ref, np.arange(n))[1])},
           "rows": {}}
    for name in FUSED + list(extra):
        p = os.path.join(PERIMG, "%s__%s.npz" % (tag, name))
        if not os.path.exists(p):
            continue
        st = load(p)
        d50 = np.array([pooled(st, d)[0] for d in idx]) - base50
        d95 = np.array([pooled(st, d)[1] for d in idx]) - base95
        out["rows"][name] = {
            "mAP50": float(pooled(st, np.arange(n))[0]),
            "mAP50_95": float(pooled(st, np.arange(n))[1]),
            "delta50": float(d50.mean()),
            "lo50": float(np.percentile(d50, 2.5)),
            "hi50": float(np.percentile(d50, 97.5)),
            "delta95": float(d95.mean()),
            "lo95": float(np.percentile(d95, 2.5)),
            "hi95": float(np.percentile(d95, 97.5))}
    return out


def report(tag, res):
    print("== %s ==" % tag)
    print("   visible source mAP50 %.4f (mAP50-95 %.4f)" %
          (res["vis"]["mAP50"], res["vis"]["mAP50_95"]))
    below = above = unres = 0
    for name, r in res["rows"].items():
        star = "*"
        if r["hi50"] < 0:
            below += 1
        elif r["lo50"] > 0:
            above += 1
        else:
            star = " "
            unres += 1
        print("   %-16s %+.4f [%+.4f,%+.4f]%s  mAP50 %.4f" %
              (name, r["delta50"], r["lo50"], r["hi50"], star, r["mAP50"]))
    print("   resolved below %d, above %d, unresolved %d" % (below, above, unres))


if __name__ == "__main__":
    s = contrasts("yolo11s_vis", draws=1000)
    report("yolo11s_vis (1000 draws)", s)
    json.dump(s, open(r"D:\CSF-main\ssgf\_restore_yolo11s.json", "w"), indent=2)

    res = {"i512": contrasts("yolo11n_vis_i512", draws=400),
           "i640": contrasts("yolo11n_vis", draws=400),
           "i960": contrasts("yolo11n_vis_i960", draws=400)}
    for k, v in res.items():
        report("yolo11n_vis @ imgsz %s (400 draws)" % k[1:], v)
    json.dump(res, open(r"D:\CSF-main\ssgf\_restore_resolution.json", "w"), indent=2)
    print("NEWDET_DONE")
