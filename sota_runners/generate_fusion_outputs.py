"""Generate the SSGF/IFCNN/DenseFuse fused-image directories used by the paper.

The paper evaluates fusion at a maximum side of 512 pixels.  This script applies
that resize once to the registered source pair, runs the selected rules and
baselines at that resolution, and writes one grayscale PNG per source stem.

Example:
    python -m sota_runners.generate_fusion_outputs \
        --root dataset/msrs/test --vis-dir vi --ir-dir ir \
        --out-root outputs/msrs --methods mean tophat probe ensemble ifcnn densefuse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, build_prior
from ssgf.models.baselines import DenseFuse, IFCNN, fuse_l1
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import find_pairs, imread, imsave


SSGF_METHODS = ("mean", "tophat", "probe", "ensemble")
BASELINE_METHODS = ("ifcnn", "densefuse")
ALL_METHODS = SSGF_METHODS + BASELINE_METHODS


def resize_pair(vis: np.ndarray, ir: np.ndarray, max_side: int):
    """Resize a registered pair without changing its aspect ratio."""
    if max_side <= 0:
        return vis, ir
    h, w = vis.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return vis, ir
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))

    def resize(image):
        encoded = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return np.asarray(Image.fromarray(encoded).resize(size, Image.BICUBIC),
                          dtype=np.float32) / 255.0

    return resize(vis), resize(ir)


def save_float(path: Path, image: np.ndarray) -> None:
    """Save a float image with the same quantization used by the paper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(str(path), np.clip(image, 0, 1))


def load_models(args, device: str):
    cfg = get_default_config()
    encoder, decoder = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae_ckpt, encoder=encoder, decoder=decoder)

    mean_rule = MeanRule(encoder, decoder, cfg, device)
    tophat_rule = SemanticPriorRule(
        encoder, decoder, cfg, build_prior("tophat"), device
    )
    probe_rule = SemanticSaliencyRule(encoder, decoder, cfg, device)
    load_ckpt(args.probe_ckpt, probe=probe_rule.probe)

    ifcnn = IFCNN().to(device).eval()
    ifcnn.load_state_dict(
        torch.load(args.ifcnn_ckpt, map_location=device, weights_only=False)
    )
    densefuse = DenseFuse().to(device).eval()
    densefuse.load_state_dict(
        torch.load(args.densefuse_ckpt, map_location=device, weights_only=False)
    )
    return mean_rule, tophat_rule, probe_rule, ifcnn, densefuse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="directory containing vis/ir subdirs")
    parser.add_argument("--vis-dir", default="vis")
    parser.add_argument("--ir-dir", default="ir")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS,
                        default=list(ALL_METHODS))
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ae-ckpt", default="data/msrs_ae.pt")
    parser.add_argument("--probe-ckpt", default="data/msrs_probe.pt")
    parser.add_argument("--ifcnn-ckpt", default="data/btl/ifcnn.pt")
    parser.add_argument("--densefuse-ckpt", default="data/btl/densefuse.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean_rule, tophat_rule, probe_rule, ifcnn, densefuse = load_models(args, device)
    pairs = find_pairs(args.root, args.vis_dir, args.ir_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"No matched source pairs under {args.root}")

    out_root = Path(args.out_root)
    for method in args.methods:
        (out_root / method).mkdir(parents=True, exist_ok=True)
    print(f"[generate] {len(pairs)} pairs, methods={','.join(args.methods)}, "
          f"max_side={args.max_side}, device={device}", flush=True)

    with torch.no_grad():
        for index, (vis_path, ir_path) in enumerate(pairs, start=1):
            stem = Path(vis_path).stem
            vis, ir = resize_pair(imread(vis_path), imread(ir_path), args.max_side)
            vis_t = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0).to(device)
            ir_t = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).to(device)

            fused = {}
            if "mean" in args.methods:
                fused["mean"] = mean_rule.fuse_pair(vis_t, ir_t)[0][0, 0].cpu().numpy()
            if "tophat" in args.methods or "ensemble" in args.methods:
                tophat = tophat_rule.fuse_pair(vis_t, ir_t)[0][0, 0].cpu().numpy()
                tophat_for_ensemble = tophat
                if "tophat" in args.methods:
                    fused["tophat"] = tophat
            if "probe" in args.methods or "ensemble" in args.methods:
                probe = probe_rule.fuse_pair(vis_t, ir_t)[0][0, 0].cpu().numpy()
                if "probe" in args.methods:
                    fused["probe"] = probe
                if "ensemble" in args.methods:
                    fused["ensemble"] = (tophat_for_ensemble + probe) / 2.0
            if "ifcnn" in args.methods:
                encoded = fuse_l1(ifcnn.encode(vis_t * 2 - 1),
                                  ifcnn.encode(ir_t * 2 - 1))
                fused["ifcnn"] = ((ifcnn.decode(encoded)[0, 0] + 1) /
                                  2).cpu().numpy()
            if "densefuse" in args.methods:
                encoded = fuse_l1(densefuse.encode(vis_t * 2 - 1),
                                  densefuse.encode(ir_t * 2 - 1))
                fused["densefuse"] = (
                    (densefuse.decode(encoded)[0, 0] + 1) / 2
                ).cpu().numpy()

            for method, image in fused.items():
                save_float(out_root / method / f"{stem}.png", image)
            if index % 25 == 0 or index == len(pairs):
                print(f"[generate] {index}/{len(pairs)}", flush=True)


if __name__ == "__main__":
    main()
