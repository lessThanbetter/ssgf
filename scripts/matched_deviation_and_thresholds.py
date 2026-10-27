"""Matched-weight-deviation comparison and confidence-threshold sensitivity."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_ap import EPS, compute_ap, load, pooled  # noqa: E402

PERIMG = r"D:\CSF-main\ssgf\_restore_perimg"
TAU = r"D:\CSF-main\ssgf\_restore_tau.json"
DETS = ["yolo11n_vis", "yolo26n_vis", "rtdetr_l_vis", "agnostic"]
SHOW = ["mean", "probe", "tophat", "ifcnn", "spgfusion"]


def matched_deviation(tau):
    """Interpolate mAP50 at common weight deviations for the three temperature rules."""
    rows = tau["rows"]
    out = {}
    targets = [0.03, 0.06, 0.12]
    for ev in ("vis", "agn"):
        curves = {}
        for rule in ("activity_only", "tophat", "probe"):
            xs = np.array([r["w_dev"] for r in rows
                           if r["evaluator"] == ev and r["rule"] == rule])
            ys = np.array([r["mAP50"] for r in rows
                           if r["evaluator"] == ev and r["rule"] == rule])
            order = np.argsort(xs)
            curves[rule] = (xs[order], ys[order])
        out[ev] = {}
        for t in targets:
            vals = {}
            for rule, (xs, ys) in curves.items():
                if xs[0] <= t <= xs[-1]:
                    vals[rule] = float(np.interp(t, xs, ys))
            out[ev][str(t)] = vals
    return out


def f1_at_threshold(st, idx, conf_min, col=0):
    tp = np.concatenate([st["tp"][i] for i in idx])
    cf = np.concatenate([st["cf"][i] for i in idx])
    gt = np.concatenate([st["gt"][i] for i in idx])
    sel = cf >= conf_min
    if sel.sum() == 0 or len(gt) == 0:
        return 0.0, 0.0, 0.0
    npos = len(gt)
    t = tp[sel, col].astype(float)
    tp_c, fp_c = t.sum(), float(sel.sum() - t.sum())
    rec = tp_c / npos
    pre = tp_c / max(tp_c + fp_c, 1e-12)
    f1 = 2 * pre * rec / max(pre + rec, 1e-12)
    return float(rec), float(pre), float(f1)


def main():
    out = {"matched_deviation": matched_deviation(json.load(open(TAU, encoding="utf-8")))}
    print("== mAP50 at matched weight deviation ==")
    for ev, by_t in out["matched_deviation"].items():
        for t, vals in by_t.items():
            print("  %-4s w_dev %.2f  %s" % (ev, float(t),
                  "  ".join("%s %.4f" % (k, v) for k, v in vals.items())))

    thr = {}
    for det in DETS:
        ref = load(os.path.join(PERIMG, "%s__vis.npz" % det))
        idx = np.arange(ref["n"])
        thr[det] = {}
        for cmin in (0.05, 0.25, 0.5):
            row = {"vis": f1_at_threshold(ref, idx, cmin)}
            for name in SHOW:
                p = os.path.join(PERIMG, "%s__%s.npz" % (det, name))
                if os.path.exists(p):
                    row[name] = f1_at_threshold(load(p), idx, cmin)
            thr[det][str(cmin)] = row
    out["thresholds"] = thr
    print("== F1 at fixed confidence thresholds ==")
    for det, by_t in thr.items():
        for cmin, row in by_t.items():
            base = row["vis"]
            print("  %-14s conf>=%.2f  vis F1 %.3f | %s" % (
                det, float(cmin), base[2],
                "  ".join("%s %+.3f" % (n, row[n][2] - base[2])
                          for n in SHOW if n in row and n != "mean") + 
                "  mean %+.3f" % (row["mean"][2] - base[2]) if "mean" in row else ""))
    json.dump(out, open(r"D:\CSF-main\ssgf\_restore_second.json", "w"), indent=2)
    print("SECOND_DONE")


if __name__ == "__main__":
    main()
