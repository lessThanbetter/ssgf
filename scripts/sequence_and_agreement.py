"""Further analyses: per-sequence deficits, the detection resolution bound, and
image-level agreement between evaluators."""
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
DETS = ["yolo11n_vis", "yolo26n_vis", "rtdetr_l_vis", "agnostic"]


def per_image_f1(st, i, col=0):
    """F1 of image i at its own maximum-F1 confidence."""
    tp = st["tp"][i]
    cf = st["cf"][i]
    gt = st["gt"][i]
    if len(tp) == 0 or len(gt) == 0:
        return 0.0, 0
    order = np.argsort(-cf)
    t = tp[order, col].astype(float)
    tpc, fpc = t.cumsum(), (1.0 - t).cumsum()
    rec = tpc / len(gt)
    pre = tpc / np.maximum(tpc + fpc, 1e-12)
    f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-12)
    k = int(np.argmax(f1))
    return float(f1[k]), int(tpc[k])


def main():
    out = {}

    # ---- per-sequence deficits under the primary detector
    ref = load(os.path.join(PERIMG, "yolo11n_vis__vis.npz"))
    stems = ref["stems"]
    seq_of = [s[:3] for s in stems]
    seqs = sorted(set(seq_of))
    per_seq = {}
    for sq in seqs:
        idx = np.array([i for i, s in enumerate(seq_of) if s == sq])
        if len(idx) < 20:
            continue
        base = pooled(ref, idx)[0]
        row = {"n": int(len(idx)), "vis_mAP50": float(base)}
        for name in FUSED:
            p = os.path.join(PERIMG, "yolo11n_vis__%s.npz" % name)
            if os.path.exists(p):
                row[name] = float(pooled(load(p), idx)[0] - base)
        per_seq[sq] = row
    out["per_sequence"] = per_seq
    print("== per-sequence deficits, primary visible-trained detector ==")
    print("%-6s %4s %8s %9s %9s %9s %9s" % ("seq", "n", "vis", "mean", "probe", "ifcnn", "spg"))
    for sq, r in sorted(per_seq.items(), key=lambda kv: -kv[1]["n"]):
        print("%-6s %4d %8.4f %+9.3f %+9.3f %+9.3f %+9.3f"
              % (sq, r["n"], r["vis_mAP50"], r.get("mean", np.nan), r.get("probe", np.nan),
                 r.get("ifcnn", np.nan), r.get("spgfusion", np.nan)))

    # ---- the detection resolution bound: bootstrap spread of each contrast
    rng = np.random.default_rng(11)
    bound = {}
    for det in DETS:
        base_st = load(os.path.join(PERIMG, "%s__vis.npz" % det))
        n = base_st["n"]
        idx_all = np.arange(n)
        draws = [rng.choice(idx_all, size=n, replace=True) for _ in range(400)]
        b = np.array([pooled(base_st, d)[0] for d in draws])
        for name in FUSED:
            p = os.path.join(PERIMG, "%s__%s.npz" % (det, name))
            if not os.path.exists(p):
                continue
            st = load(p)
            d = np.array([pooled(st, dd)[0] for dd in draws]) - b
            bound["%s|%s" % (det, name)] = float(d.std(ddof=1))
    sds = np.array(list(bound.values()))
    out["resolution_bound"] = {
        "median_sd": float(np.median(sds)),
        "p90_sd": float(np.percentile(sds, 90)),
        "median_detectable": float(2.8 * np.median(sds)),
        "bound_by_contrast": bound}
    print("== detection resolution ==")
    print("  median bootstrap SD of a contrast: %.4f mAP50" % np.median(sds))
    print("  median smallest resolvable deficit (2.8 sd): %.4f" % (2.8 * np.median(sds)))
    print("  largest SD: %.4f  smallest: %.4f" % (sds.max(), sds.min()))
    big = sorted(bound.items(), key=lambda kv: -kv[1])[:4]
    for k, v in big:
        print("    widest:", k, "%.4f" % v)

    # ---- image-level agreement between evaluators
    imgs = {}
    for det in DETS:
        v = load(os.path.join(PERIMG, "%s__vis.npz" % det))
        s_path = os.path.join(PERIMG, "%s__spgfusion.npz" % det)
        if not os.path.exists(s_path):
            continue
        s = load(s_path)
        d = np.array([per_image_f1(s, i)[0] - per_image_f1(v, i)[0] for i in range(v["n"])])
        imgs[det] = d
    agree = {}
    for a in imgs:
        for b in imgs:
            if a >= b:
                continue
            x, y = imgs[a], imgs[b]
            same = float(np.mean(np.sign(x) == np.sign(y)))
            both = float(np.mean((x < 0) & (y < 0)))
            corr = float(np.corrcoef(x, y)[0, 1])
            agree["%s|%s" % (a, b)] = {"sign_agreement": same, "both_deficit": both,
                                       "corr": corr}
    out["image_level_agreement"] = agree
    print("== image-level agreement on the SPGFusion contrast (per-image F1) ==")
    for k, v in sorted(agree.items()):
        print("  %-28s sign agreement %.3f  both negative %.3f  corr %+.2f"
              % (k, v["sign_agreement"], v["both_deficit"], v["corr"]))

    json.dump(out, open(r"D:\CSF-main\ssgf\_restore_more.json", "w"), indent=2)
    print("MORE_DONE")


if __name__ == "__main__":
    main()
