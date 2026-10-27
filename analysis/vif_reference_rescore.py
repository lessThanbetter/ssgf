"""Rescore every fused variant under the reported index and the reference pixel VIF.

Self-contained: it does not import the repository, so it can run on the server
without touching the installed package. Both indices are implemented here.

Reported index (what the paper's tables use)
  four scales, Gaussian windows 17 / 9 / 5 / 3, width 2**(s-1), no decimation
  between scales, noise variance 2.0 in the units of the input, which is [0, 1]

Reference construction
  same windows, width = window / 5, decimation by two between scales, noise
  variance 2 / 255**2 so that it is expressed in the units of the input

Both indices are the mean of the two single-reference values, against the visible
and the infrared source, as the paper defines its fusion index.

All three images of a pair are resized with the same convention before scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import scipy.ndimage as ndi
from PIL import Image


def _gauss_kernel(size, sigma):
    ax = np.arange(size) - (size // 2)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def _gray(x):
    return np.asarray(x, dtype=np.float32)


def _read(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _resize(x, shape):
    h, w = shape
    im = Image.fromarray((np.clip(x, 0, 1) * 255).astype("uint8"))
    return np.asarray(im.resize((w, h)), dtype=np.float32) / 255.0


def _load_same_size(vis_p, ir_p, fused_p, max_side, target=None):
    """Return the three images on one common grid.

    The visible source defines the registered pair's geometry.  This matters for
    RoadScene, whose high-resolution visible crop and 640 by 512 infrared frame have
    different dimensions but the same field of view.  Cropping both to the smaller
    rectangle would silently discard part of the scene.  The infrared source and
    fused output are therefore resampled to the (optionally size-capped) visible
    grid.  If ``target`` is given, that shape is used for all three instead.
    """
    vis, ir, fu = _read(vis_p), _read(ir_p), _read(fused_p)
    h, w = vis.shape
    if target is not None:
        vis, ir, fu = _resize(vis, target), _resize(ir, target), _resize(fu, target)
    elif max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        new = (max(1, int(round(h * s))), max(1, int(round(w * s))))
        vis, ir, fu = _resize(vis, new), _resize(ir, new), _resize(fu, new)
    else:
        vis, ir, fu = vis, _resize(ir, (h, w)), _resize(fu, (h, w))
    return vis, ir, fu


def vif(ref, dist, width_mode, decimate, sigma_nsq):
    ref, dist = _gray(ref).copy(), _gray(dist).copy()
    num = den = 0.0
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        sigma = (n / 5.0) if width_mode == "canonical" else (2.0 ** (scale - 1))
        g = _gauss_kernel(n, sigma)
        if scale > 1:
            ref = ndi.convolve(ref, g, mode="nearest")
            dist = ndi.convolve(dist, g, mode="nearest")
            if decimate:
                ref, dist = ref[::2, ::2], dist[::2, ::2]
        mu1 = ndi.convolve(ref, g, mode="nearest")
        mu2 = ndi.convolve(dist, g, mode="nearest")
        s11 = ndi.convolve(ref * ref, g, mode="nearest") - mu1 * mu1
        s22 = ndi.convolve(dist * dist, g, mode="nearest") - mu2 * mu2
        s12 = ndi.convolve(ref * dist, g, mode="nearest") - mu1 * mu2
        gm = np.clip(s12 / (s11 + 1e-9), 0, 1)
        sv = s22 - gm * s12
        sv[sv < 0] = 0
        num += np.sum(np.log10(1 + gm ** 2 * s11 / (sv + sigma_nsq)))
        den += np.sum(np.log10(1 + s11 / sigma_nsq))
    return float(num / (den + 1e-9))


def reported(vis, ir, fu):
    return 0.5 * (vif(vis, fu, "reported", False, 2.0)
                  + vif(ir, fu, "reported", False, 2.0))


def reference(vis, ir, fu):
    s = 2.0 / (255.0 ** 2)
    return 0.5 * (vif(vis, fu, "canonical", True, s)
                  + vif(ir, fu, "canonical", True, s))


def reported_vis(vis, ir, fu):
    del ir
    return vif(vis, fu, "reported", False, 2.0)


def reference_vis(vis, ir, fu):
    del ir
    return vif(vis, fu, "canonical", True, 2.0 / (255.0 ** 2))


def width_only(vis, ir, fu):
    return 0.5 * (vif(vis, fu, "canonical", False, 2.0)
                  + vif(ir, fu, "canonical", False, 2.0))


def find(d, stem):
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", required=True)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--fused", action="append", required=True)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max_side", type=int, default=0)
    ap.add_argument("--target_hw", default="", help="HxW common grid for all three images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields", nargs="+",
                    choices=("reported", "reported_vis", "width_only",
                             "reference", "reference_vis"),
                    default=("reported", "width_only", "reference"),
                    help="metrics to compute (default: all three)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        rng = np.random.default_rng(0)
        a = rng.random((256, 256)).astype(np.float32)
        print("self, reported ", round(vif(a, a, "reported", False, 2.0), 6))
        print("self, reference", round(vif(a, a, "canonical", True, 2 / 255 ** 2), 6))

    variants = []
    for spec in args.fused:
        name, d = spec.split("=", 1)
        variants.append((name, d))

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.vis))
    target = None
    if args.target_hw:
        th, tw = args.target_hw.lower().split("x")
        target = (int(th), int(tw))
    keep = []
    for s in stems:
        if find(args.vis, s) and find(args.ir, s) and all(find(d, s) for _, d in variants):
            keep.append(s)
        if args.limit > 0 and len(keep) >= args.limit:
            break
    print("[rescore] usable stems:", len(keep), "max_side:", args.max_side, flush=True)

    fields = tuple(dict.fromkeys(args.fields))
    acc = {n: {field: [] for field in fields} for n, _ in variants}
    metrics = {
        "reported": reported,
        "reported_vis": reported_vis,
        "width_only": width_only,
        "reference": reference,
        "reference_vis": reference_vis,
    }
    for i, stem in enumerate(keep):
        for name, d in variants:
            vis, ir, fu = _load_same_size(find(args.vis, stem), find(args.ir, stem),
                                          find(d, stem), args.max_side, target)
            for field in fields:
                acc[name][field].append(metrics[field](vis, ir, fu))
        print("[rescore] %d/%d %s" % (i + 1, len(keep), stem), flush=True)

    rows = []
    columns = ("variant", "n") + fields
    for name, _ in variants:
        row = {"variant": name, "n": len(acc[name][fields[0]])}
        for field in fields:
            row[field] = float(np.mean(acc[name][field]))
        rows.append(row)

    out_parent = os.path.dirname(args.out)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(args.out + ".json", "w") as fh:
        json.dump({"limit": args.limit, "max_side": args.max_side,
                   "fields": fields, "rows": rows}, fh, indent=2)

    print()
    headers = [field.replace("_", "-") for field in fields]
    header_widths = [max(10, len(header)) for header in headers]
    print("%-12s %5s %s" % (
        "variant", "n",
        " ".join(f"{header:>{width}}" for header, width
                 in zip(headers, header_widths)),
    ))
    for r in rows:
        print("%-12s %5d %s" % (
            r["variant"], r["n"],
            " ".join(f"{r[field]:>{width}.4f}" for field, width
                     in zip(fields, header_widths)),
        ))
    for field in fields:
        order = [r["variant"] for r in sorted(rows, key=lambda x: -x[field])]
        span = max(r[field] for r in rows) - min(r[field] for r in rows)
        print("ranking %-11s: %s   (span %.4f)" %
              (field, " > ".join(order), span))
    print("RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
