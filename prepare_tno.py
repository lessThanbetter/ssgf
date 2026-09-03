"""Build a registered VIS/IR paired dataset from the TNO FEL sequences.

TNO stores pairs as ``####v.bmp`` (visible) / ``####i.bmp`` (thermal) inside
``<sequence>/visual`` and ``<sequence>/thermal``. This script pairs them by the
numeric stem and writes ``dest/vis`` + ``dest/ir`` as PNGs.

Usage:
    python prepare_tno.py --tno C:/Users/.../TNO_Image_Fusion_Dataset \
        --dest D:/CSF-main/ssgf/data/tno_trial
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
from PIL import Image


def collect_sequences(tno_root: str) -> list[tuple[str, str]]:
    """Return ``(visual_dir, thermal_dir)`` for every sequence with both."""
    out = []
    for root, dirs, _ in os.walk(tno_root):
        if "visual" in dirs and "thermal" in dirs:
            out.append((os.path.join(root, "visual"), os.path.join(root, "thermal")))
    return sorted(out)


def pair_by_stem(vis_dir: str, thr_dir: str) -> list[tuple[str, str]]:
    vis = {}
    for f in os.listdir(vis_dir):
        m = re.match(r"(\d+)[vV]", os.path.splitext(f)[0])
        if m:
            vis.setdefault(int(m.group(1)), f)
    thr = {}
    for f in os.listdir(thr_dir):
        m = re.match(r"(\d+)[iI]", os.path.splitext(f)[0])
        if m:
            thr.setdefault(int(m.group(1)), f)
    keys = sorted(set(vis) & set(thr))
    return [(os.path.join(vis_dir, vis[k]), os.path.join(thr_dir, thr[k])) for k in keys]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tno", required=True, help="path to TNO_Image_Fusion_Dataset")
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.dest, "vis"), exist_ok=True)
    os.makedirs(os.path.join(args.dest, "ir"), exist_ok=True)
    total = 0
    for vis_dir, thr_dir in collect_sequences(args.tno):
        pairs = pair_by_stem(vis_dir, thr_dir)
        for vis_path, thr_path in pairs:
            stem = os.path.splitext(os.path.basename(vis_path))[0][:-1]
            name = f"{os.path.basename(os.path.dirname(vis_dir))}_{stem}"  # <scene>_<num>
            for src, dst_dir in ((vis_path, "vis"), (thr_path, "ir")):
                im = Image.open(src).convert("L")
                a = np.asarray(im, dtype=np.float32) / 255.0
                Image.fromarray((a * 255).astype("uint8")).save(
                    os.path.join(args.dest, dst_dir, f"{name}.png")
                )
            total += 1
    print(f"[tno] wrote {total} pairs to {args.dest}/vis + {args.dest}/ir")


if __name__ == "__main__":
    main()
