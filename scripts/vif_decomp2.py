"""Decompose the reported index's three departures, on the Table S6 protocol.

Same images, grid and eight variants as Supplementary Table S6 (the recorded values
are reproduced by the first three columns), with two further columns that change one
departure at a time. The Gaussian kernels here are separable and are applied as two
one-dimensional convolutions, which is algebraically identical to the two-dimensional
kernel of the released script and about an order of magnitude faster.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.ndimage as ndi
from PIL import Image


def _kernel1d(size, sigma):
    ax = np.arange(size) - (size // 2)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _conv(x, k):
    return ndi.convolve1d(ndi.convolve1d(x, k, axis=0, mode="nearest"),
                          k, axis=1, mode="nearest")


def _read(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _resize(x, shape):
    h, w = shape
    im = Image.fromarray((np.clip(x, 0, 1) * 255).astype("uint8"))
    return np.asarray(im.resize((w, h)), dtype=np.float32) / 255.0


def _load(vis_p, ir_p, fused_p, target):
    vis, ir, fu = _read(vis_p), _read(ir_p), _read(fused_p)
    h = min(vis.shape[0], ir.shape[0])
    w = min(vis.shape[1], ir.shape[1])
    vis, ir = vis[:h, :w], ir[:h, :w]
    fu = _resize(fu, (h, w))
    if target is not None:
        vis, ir, fu = _resize(vis, target), _resize(ir, target), _resize(fu, target)
    return vis, ir, fu


def vif(ref, dist, width_mode, decimate, sigma_nsq):
    ref, dist = ref.copy(), dist.copy()
    num = den = 0.0
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        sigma = (n / 5.0) if width_mode == "canonical" else (2.0 ** (scale - 1))
        k = _kernel1d(n, sigma)
        if scale > 1:
            ref = _conv(ref, k)
            dist = _conv(dist, k)
            if decimate:
                ref, dist = ref[::2, ::2], dist[::2, ::2]
        mu1 = _conv(ref, k)
        mu2 = _conv(dist, k)
        s11 = _conv(ref * ref, k) - mu1 * mu1
        s22 = _conv(dist * dist, k) - mu2 * mu2
        s12 = _conv(ref * dist, k) - mu1 * mu2
        gm = np.clip(s12 / (s11 + 1e-9), 0, 1)
        sv = s22 - gm * s12
        sv[sv < 0] = 0
        num += np.sum(np.log10(1 + gm ** 2 * s11 / (sv + sigma_nsq)))
        den += np.sum(np.log10(1 + s11 / sigma_nsq))
    return float(num / (den + 1e-9))


REF = 2.0 / (255.0 ** 2)
CONFIGS = [
    ("reported", "reported", False, 2.0),
    ("width_only", "canonical", False, 2.0),
    ("decimate_only", "reported", True, 2.0),
    ("noise_only", "reported", False, REF),
    ("reference", "canonical", True, REF),
]


def score(vis, ir, fu, cfg):
    _n, width_mode, decimate, sigma = cfg
    return 0.5 * (vif(vis, fu, width_mode, decimate, sigma)
                  + vif(ir, fu, width_mode, decimate, sigma))


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
    ap.add_argument("--target_hw", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target = None
    if args.target_hw:
        th, tw = args.target_hw.lower().split("x")
        target = (int(th), int(tw))

    variants = [s.split("=", 1) for s in args.fused]
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.vis))
    keep = []
    for s in stems:
        if find(args.vis, s) and find(args.ir, s) and all(find(d, s) for _, d in variants):
            keep.append(s)
        if len(keep) >= args.limit:
            break
    print("[decomp] usable stems:", len(keep), "target:", target, flush=True)

    acc = {n: {c[0]: [] for c in CONFIGS} for n, _ in variants}
    for i, stem in enumerate(keep):
        for name, d in variants:
            vis, ir, fu = _load(find(args.vis, stem), find(args.ir, stem),
                                find(d, stem), target)
            for cfg in CONFIGS:
                acc[name][cfg[0]].append(score(vis, ir, fu, cfg))
        print("[decomp] %d/%d" % (i + 1, len(keep)), flush=True)

    rows = []
    for name, _ in variants:
        row = {"variant": name, "n": len(acc[name]["reported"])}
        row.update({k: float(np.mean(v)) for k, v in acc[name].items()})
        rows.append(row)

    json.dump({"limit": args.limit, "target_hw": args.target_hw, "rows": rows},
              open(args.out + ".json", "w"), indent=2)
    cols = [c[0] for c in CONFIGS]
    print()
    print("%-12s %4s %s" % ("variant", "n", " ".join("%13s" % c for c in cols)))
    for r in rows:
        print("%-12s %4d %s" % (r["variant"], r["n"], " ".join("%13.4f" % r[c] for c in cols)))
    for c in cols:
        order = [r["variant"] for r in sorted(rows, key=lambda x: -x[c])]
        span = max(r[c] for r in rows) - min(r[c] for r in rows)
        print("ranking %-14s %s   (span %.4f)" % (c, " > ".join(order), span))
    print("VIF_DECOMP_DONE", flush=True)


if __name__ == "__main__":
    main()
