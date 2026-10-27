"""Local reimplementation of the detector library's average precision.

Identical in structure to `ultralytics.utils.metrics.ap_per_class` and its
`compute_ap`: sort detections by confidence, accumulate false positives and true
positives per class, then integrate the interpolated precision-recall curve over the
101-point recall grid. Used here so that the per-image statistics can be analysed
without the server, and validated against the values the server produced.
"""
import numpy as np

EPS = 1e-16


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy < 2.0
        trapezoid = np.trapz
    return float(trapezoid(np.interp(x, mrec, mpre), x))


def ap_per_class(tp, conf, pred_cls, target_cls):
    """tp is (n, 10) bool; returns (mAP50, mAP50-95) over the classes present."""
    if len(tp) == 0 or len(target_cls) == 0:
        return 0.0, 0.0
    order = np.argsort(-conf)
    tp, pred_cls = tp[order], pred_cls[order]
    classes, counts = np.unique(target_cls, return_counts=True)
    aps = []
    for c, n_l in zip(classes, counts):
        sel = pred_cls == c
        if sel.sum() == 0:
            continue
        fpc = (1 - tp[sel]).cumsum(0)
        tpc = tp[sel].cumsum(0)
        recall = tpc / (n_l + EPS)
        precision = tpc / (tpc + fpc + EPS)
        aps.append([compute_ap(recall[:, j], precision[:, j]) for j in range(tp.shape[1])])
    if not aps:
        return 0.0, 0.0
    ap = np.asarray(aps)          # (n_classes, 10)
    return float(ap[:, 0].mean()), float(ap.mean())


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
    return {
        "n": n,
        "stems": list(z["stems"]),
        "tp": [tp[bounds[i]:bounds[i + 1]] for i in range(n)],
        "cf": [conf[bounds[i]:bounds[i + 1]] for i in range(n)],
        "cl": [pcls[bounds[i]:bounds[i + 1]] for i in range(n)],
        "gt": [tcls[cuts[i]:cuts[i + 1]] for i in range(n)],
        "tp_all": tp,
        "cf_all": conf,
        "cl_all": pcls,
        "gt_all": tcls,
        "det_counts": np.diff(bounds),
        "det_offsets": bounds[:-1],
        "gt_counts": n_gt,
        "gt_offsets": cuts[:-1],
    }


def pooled(st, idx):
    if len(idx) == 0:
        return 0.0, 0.0
    if "det_offsets" in st:
        def gather(values, offsets_name, counts_name):
            offsets = st[offsets_name][idx]
            counts = st[counts_name][idx]
            total = int(counts.sum())
            if total == 0:
                return values[:0]
            group_starts = np.repeat(offsets, counts)
            starts = np.repeat(np.cumsum(counts) - counts, counts)
            positions = group_starts + np.arange(total) - starts
            return values[positions]

        tp = gather(st["tp_all"], "det_offsets", "det_counts")
        cf = gather(st["cf_all"], "det_offsets", "det_counts")
        cl = gather(st["cl_all"], "det_offsets", "det_counts")
        gt = gather(st["gt_all"], "gt_offsets", "gt_counts")
        return ap_per_class(tp, cf, cl, gt)
    tp = np.concatenate([st["tp"][i] for i in idx])
    cf = np.concatenate([st["cf"][i] for i in idx])
    cl = np.concatenate([st["cl"][i] for i in idx])
    gt = np.concatenate([st["gt"][i] for i in idx])
    return ap_per_class(tp, cf, cl, gt)
