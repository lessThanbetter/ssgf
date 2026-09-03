"""Generate U2-Net pseudo-label masks for a VIS/IR pair dataset.

Runs on the server (GPU). Writes ``out/vis/<name>.png`` and ``out/ir/<name>.png``
so the probe can be distilled with ``train --mode probe``.

Usage:
    python gen_masks.py --root /root/autodl-tmp/ssgf/data/tno_trial \
        --prior /root/autodl-tmp/ssgf/third_party/u2netp.onnx \
        --out /root/autodl-tmp/ssgf/data/tno_masks
"""

from __future__ import annotations

import argparse
import os

from ssgf.models.semantic_prior import build_prior
from ssgf.utils.io import find_pairs, imread, imsave


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--vis_dir", default="vis")
    ap.add_argument("--ir_dir", default="ir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prior = build_prior(args.prior)
    pairs = find_pairs(args.root, args.vis_dir, args.ir_dir)
    print(f"[masks] generating {len(pairs) * 2} masks ...")
    n = 0
    for vis_path, ir_path in pairs:
        name = os.path.splitext(os.path.basename(vis_path))[0]
        for src_path, mod in ((vis_path, "vis"), (ir_path, "ir")):
            mask = prior(imread(src_path))
            imsave(os.path.join(args.out, mod, f"{name}.png"), mask)
            n += 1
    print(f"[masks] wrote {n} masks to {args.out}/vis + {args.out}/ir")


if __name__ == "__main__":
    main()
