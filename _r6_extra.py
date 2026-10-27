"""Additional analyses for the extended manuscript.

1. rank correlation between fidelity values and the downstream contrast,
2. the mAP against the IoU criterion, per input and evaluator,
3. the operating point behind the deficits (recall, precision, false positives),
4. multiplicity correction and draw-count stability of the 42 contrasts.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ap_iou import pooled_iou, operating_point  # noqa: E402
from _local_ap import load, pooled  # noqa: E402

PERIMG = r"D:\CSF-main\ssgf\_restore_perimg"
AGG = r"D:\CSF-main\ssgf\_deposit\ssgf-llvip-fusion-detection-v1\data\aggregate"
FID = r"D:\CSF-main\ssgf\_remote_jobs\results\j1_fidelity.json"

FUSED = ["mean", "activity_only", "tophat", "probe", "tophat_permuted", "probe_permuted",
         "tophat_wshuffle", "probe_wshuffle", "ensemble", "ifcnn", "densefuse", "csf",
         "piafusion", "spgfusion"]
DETS = {"yolo11n_vis": "contrasts_yolo11n_colour.json",
        "yolo26n_vis": "contrasts_yolo26n_colour.json",
        "rtdetr_l_vis": "contrasts_rtdetr_l_colour.json",
        "agnostic": "contrasts_agnostic_colour.json"}
IOU = np.round(np.linspace(0.5, 0.95, 10), 2)


def spearman(x, y):
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        out = np.zeros(len(v), float)
        i = 0
        sv = np.sort(v)
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            idx = np.where(v == sv[i])[0]
            out[idx] = (i + j) / 2.0
            i = j + 1
        return out
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def two_sided_p_from_draws(draws):
    d = np.asarray(draws, float)
    pos = float((d > 0).mean())
    neg = float((d < 0).mean())
    return 2.0 * min(pos, neg), pos, neg


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, float)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def main():
    out = {}
    fid = json.load(open(FID, encoding="utf-8"))
    vif_ref = {r["variant"]: r["reference"] for r in
               json.load(open(os.path.join(AGG, "vif_decomp.json"), encoding="utf-8"))["rows"]}

    # ---- 1. rank correlation between fidelity and the downstream contrast
    corr = {}
    for det, fname in DETS.items():
        d = json.load(open(os.path.join(AGG, fname), encoding="utf-8"))
        rows = [r for r in FUSED if r in d and r in fid]
        vif = [fid[r]["VIF_vis"] for r in rows]
        ss = [fid[r]["SSIM_vis"] for r in rows]
        en = [fid[r]["EN"] for r in rows]
        dl = [d[r]["delta50"] for r in rows]
        corr[det] = {"n": len(rows), "VIF": spearman(vif, dl), "SSIM_vis": spearman(ss, dl),
                     "EN": spearman(en, dl)}
    rows_ref = [r for r in vif_ref if r in fid and r in
                json.load(open(os.path.join(AGG, DETS["yolo11n_vis"]), encoding="utf-8"))]
    if len(rows_ref) >= 4:
        d = json.load(open(os.path.join(AGG, DETS["yolo11n_vis"]), encoding="utf-8"))
        corr["reference_vif_yolo11n"] = {
            "n": len(rows_ref),
            "VIF": spearman([vif_ref[r] for r in rows_ref], [d[r]["delta50"] for r in rows_ref])}
    out["rank_correlation"] = corr
    print("== rank correlation between fidelity and the colour-scored contrast ==")
    for k, v in corr.items():
        print("  %-24s n=%2d  VIF rho %+.2f  SSIM %s  EN %s"
              % (k, v["n"], v["VIF"],
                 ("%+.2f" % v["SSIM_vis"]) if "SSIM_vis" in v else "-",
                 ("%+.2f" % v["EN"]) if "EN" in v else "-"))

    # ---- 2. criterion sweep: mAP at each IoU threshold, and the contrast against vis
    crit = {}
    for det in DETS:
        p_vis = os.path.join(PERIMG, "%s__vis.npz" % det)
        if not os.path.exists(p_vis):
            continue
        ref = load(p_vis)
        idx = np.arange(ref["n"])
        base = pooled_iou(ref, idx)
        rows = {}
        for r in FUSED:
            q = os.path.join(PERIMG, "%s__%s.npz" % (det, r))
            if not os.path.exists(q):
                continue
            rows[r] = (pooled_iou(load(q), idx) - base).tolist()
        crit[det] = {"vis": base.tolist(), "contrasts": rows}
    out["criterion"] = crit
    print("== mAP50 and mAP50-95 of the visible source per evaluator ==")
    for det, v in crit.items():
        print("  %-14s mAP50 %.4f  mAP75 %.4f  mAP50-95 %.4f"
              % (det, v["vis"][0], v["vis"][5], float(np.mean(v["vis"]))))
    print("== SPGFusion contrast by criterion (IoU 0.50 to 0.95) ==")
    for det, v in crit.items():
        if "spgfusion" in v["contrasts"]:
            print("  %-14s %s" % (det, " ".join("%+.3f" % x for x in v["contrasts"]["spgfusion"])))

    # ---- 3. operating point: recall and precision at maximum F1, IoU 0.5
    op = {}
    for det in DETS:
        p_vis = os.path.join(PERIMG, "%s__vis.npz" % det)
        if not os.path.exists(p_vis):
            continue
        ref = load(p_vis)
        idx = np.arange(ref["n"])
        rows = {"vis": operating_point(ref, idx)}
        for r in FUSED:
            q = os.path.join(PERIMG, "%s__%s.npz" % (det, r))
            if os.path.exists(q):
                rows[r] = operating_point(load(q), idx)
        op[det] = rows
    out["operating_point"] = op
    print("== operating point at maximum F1 (IoU 0.5) ==")
    for det, rows in op.items():
        base = rows["vis"]
        print("  %-14s visible: recall %.3f precision %.3f F1 %.3f  missed %d of %d"
              % (det, base["recall"], base["precision"], base["f1"],
                 base["n_pos"] - base["tp"], base["n_pos"]))
        for r in ("spgfusion", "ifcnn", "probe", "mean"):
            if r in rows:
                c = rows[r]
                print("      %-10s recall %.3f (%+.3f)  precision %.3f (%+.3f)  missed %d"
                      % (r, c["recall"], c["recall"] - base["recall"], c["precision"],
                         c["precision"] - base["precision"], c["n_pos"] - c["tp"]))

    # ---- 4. multiplicity and draw-count stability of the 42 contrasts
    rng = np.random.default_rng(7)
    boot = {}
    pvals = {}
    for det, fname in list(DETS.items())[:3]:
        ref = load(os.path.join(PERIMG, "%s__vis.npz" % det))
        n = ref["n"]
        idx_all = np.arange(n)
        draws = [rng.choice(idx_all, size=n, replace=True) for _ in range(1000)]
        base = np.array([pooled(ref, d)[0] for d in draws])
        for r in FUSED:
            q = os.path.join(PERIMG, "%s__%s.npz" % (det, r))
            if not os.path.exists(q):
                continue
            st = load(q)
            cur = np.array([pooled(st, d)[0] for d in draws])
            ds = cur - base
            key = "%s|%s" % (det, r)
            boot[key] = ds
            pvals[key] = two_sided_p_from_draws(ds)[0]
    keys = list(pvals)
    q = bh([pvals[k] for k in keys])
    n_sig05 = int((q < 0.05).sum())
    n_sig01 = int((q < 0.01).sum())
    print("== multiplicity over the %d contrasts of the three visible-trained architectures ==" % len(keys))
    print("  q < 0.05: %d   q < 0.01: %d" % (n_sig05, n_sig01))
    worst = sorted(zip(q, keys))[-4:]
    for qq, k in worst:
        print("  largest q: %-28s q=%.3f  p=%.3f" % (k, qq, pvals[k]))
    stab = {}
    for nd in (100, 200, 400, 1000):
        resolved = 0
        for k in keys:
            ds = boot[k][:nd]
            lo, hi = np.percentile(ds, [2.5, 97.5])
            if lo > 0 or hi < 0:
                resolved += 1
        stab[nd] = resolved
    print("  contrasts resolved at 100/200/400/1000 draws:",
          [stab[d] for d in (100, 200, 400, 1000)], "of", len(keys))
    out["multiplicity"] = {"n_contrasts": len(keys), "q05": n_sig05, "q01": n_sig01,
                           "max_q": float(max(q)), "resolved_by_draws": stab}

    json.dump(out, open(r"D:\CSF-main\ssgf\_restore_extra.json", "w"), indent=2)
    print("EXTRA_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
