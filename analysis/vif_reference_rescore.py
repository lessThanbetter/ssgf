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

    The registered sources are taken as the reference geometry. The fused image is
    resized to that geometry, and then all three are optionally resized by a common
    factor so that the longest side is ``max_side``. If ``target`` is given, that
    shape is used for all three instead, which is how the published baselines are
    compared here: their fused outputs survive only at 400 by 512.
    """
    vis, ir, fu = _read(vis_p), _read(ir_p), _read(fused_p)
    h = min(vis.shape[0], ir.shape[0])
    w = min(vis.shape[1], ir.shape[1])
    vis, ir = vis[:h, :w], ir[:h, :w]
    fu = _resize(fu, (h, w))
    if target is not None:
        vis, ir, fu = _resize(vis, target), _resize(ir, target), _resize(fu, target)
    elif max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        new = (max(1, int(round(h * s))), max(1, int(round(w * s))))
        vis, ir, fu = _resize(vis, new), _resize(ir, new), _resize(fu, new)
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
        if len(keep) >= args.limit:
            break
    print("[rescore] usable stems:", len(keep), "max_side:", args.max_side, flush=True)

    acc = {n: {"reported": [], "width_only": [], "reference": []} for n, _ in variants}
    for i, stem in enumerate(keep):
        for name, d in variants:
            vis, ir, fu = _load_same_size(find(args.vis, stem), find(args.ir, stem),
                                          find(d, stem), args.max_side, target)
            acc[name]["reported"].append(reported(vis, ir, fu))
            acc[name]["width_only"].append(width_only(vis, ir, fu))
            acc[name]["reference"].append(reference(vis, ir, fu))
        print("[rescore] %d/%d %s" % (i + 1, len(keep), stem), flush=True)

    rows = []
    for name, _ in variants:
        m = {k: float(np.mean(v)) for k, v in acc[name].items()}
        rows.append({"variant": name, "n": len(acc[name]["reported"]),
                     "reported": m["reported"], "width_only": m["width_only"],
                     "reference": m["reference"]})

    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "n", "reported", "width_only", "reference"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(args.out + ".json", "w") as fh:
        json.dump({"limit": args.limit, "max_side": args.max_side, "rows": rows}, fh, indent=2)

    print()
    print("%-12s %5s %10s %11s %11s" % ("variant", "n", "reported", "width-only", "reference"))
    for r in rows:
        print("%-12s %5d %10.4f %11.4f %11.4f"
              % (r["variant"], r["n"], r["reported"], r["width_only"], r["reference"]))
    for key in ("reported", "width_only", "reference"):
        order = [r["variant"] for r in sorted(rows, key=lambda x: -x[key])]
        span = max(r[key] for r in rows) - min(r[key] for r in rows)
        print("ranking %-11s: %s   (span %.4f)" % (key, " > ".join(order), span))
    print("RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
