"""Generate the remaining validation candidates for selective fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.fusion import BaseRule, ModalityCSRule
from ssgf.models import Decoder, Encoder
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread
from sota_runners.generate_fusion_outputs import find_pairs, resize_pair


class ActivityOnlyRule(BaseRule):
    def __init__(self, encoder, decoder, cfg, device="cuda", temperature=0.3):
        super().__init__(encoder, decoder, cfg, device)
        self.temperature = temperature

    def saliency(self, vis_n, ir_n):
        vis_activity = self.encoder(vis_n).abs().sum(dim=1)
        ir_activity = self.encoder(ir_n).abs().sum(dim=1)
        maximum = torch.max(vis_activity.max(), ir_activity.max()) + 1e-8
        e_vis = torch.exp(vis_activity / maximum / self.temperature)
        e_ir = torch.exp(ir_activity / maximum / self.temperature)
        weight_vis = e_vis / (e_vis + e_ir)
        return weight_vis.unsqueeze(0), (1.0 - weight_vis).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/selective_fusion/validation_sources")
    parser.add_argument("--out", default="results/selective_fusion/validation_candidates")
    parser.add_argument("--ae-ckpt", default="data/msrs_ae.pt")
    parser.add_argument("--probe-ckpt", default="data/msrs_probe.pt")
    parser.add_argument("--csf-ckpt", default="data/msrs_csf.pt")
    parser.add_argument("--max-side", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    encoder, decoder = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae_ckpt, encoder=encoder, decoder=decoder)
    activity = ActivityOnlyRule(encoder, decoder, cfg, device)
    csf = ModalityCSRule(encoder, decoder, cfg, device)
    load_ckpt(args.csf_ckpt, classifier=csf.scorer)

    pairs = find_pairs(args.root, "visible", "infrared")
    output = Path(args.out)
    for method in ("activity_only", "csf"):
        (output / method).mkdir(parents=True, exist_ok=True)

    for index, (visible_path, infrared_path) in enumerate(pairs, 1):
        stem = Path(visible_path).stem
        visible, infrared = resize_pair(
            imread(visible_path), imread(infrared_path), args.max_side
        )
        visible_t = torch.from_numpy(visible).unsqueeze(0).unsqueeze(0).to(device)
        infrared_t = torch.from_numpy(infrared).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            activity_image = activity.fuse_pair(visible_t, infrared_t)[0][0, 0]
        csf_image = csf.fuse_pair(visible_t, infrared_t)[0][0, 0]
        for method, tensor in (
            ("activity_only", activity_image),
            ("csf", csf_image),
        ):
            image = np.clip(tensor.detach().cpu().numpy(), 0.0, 1.0)
            Image.fromarray((image * 255).astype(np.uint8)).save(
                output / method / f"{stem}.png"
            )
        if index % 25 == 0 or index == len(pairs):
            print(f"generated {index}/{len(pairs)}", flush=True)


if __name__ == "__main__":
    main()
