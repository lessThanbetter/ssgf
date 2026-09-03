"""Prepare a paired VIS/IR dataset from common public benchmarks.

Works out-of-the-box for datasets that already ship separate ``vis`` and ``ir``
folders (RoadScene, MSRS, M3FD). Pairs are matched by basename and written to
``dest/vis`` + ``dest/ir``. Optionally down-sample to a max side length so the
network sees images of a manageable size.

Usage:
    python prepare_data.py --vis_root /data/roadscene/vis --ir_root /data/roadscene/ir \
        --dest /data/ssgf/roadscene --max_side 512
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from ssgf.utils.io import find_pairs, imread, imsave


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis_root", required=True)
    ap.add_argument("--ir_root", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--max_side", type=int, default=0, help="0 = keep original size")
    args = ap.parse_args()

    pairs = find_pairs(os.path.dirname(args.vis_root), os.path.basename(args.vis_root), os.path.basename(args.ir_root))
    print(f"[prep] {len(pairs)} pairs")
    for vis_path, ir_path in pairs:
        name = os.path.splitext(os.path.basename(vis_path))[0]
        vis = imread(vis_path)
        ir = imread(ir_path)
        if args.max_side > 0:
            h, w = vis.shape
            scale = args.max_side / max(h, w)
            if scale < 1:
                new = (int(w * scale), int(h * scale))
                vis = np.asarray(Image.fromarray((vis * 255).astype("uint8")).resize(new)) / 255.0
                ir = np.asarray(Image.fromarray((ir * 255).astype("uint8")).resize(new)) / 255.0
        imsave(os.path.join(args.dest, "vis", f"{name}.png"), vis)
        imsave(os.path.join(args.dest, "ir", f"{name}.png"), ir)
    print(f"[prep] written to {args.dest}/vis + {args.dest}/ir")


if __name__ == "__main__":
    main()
