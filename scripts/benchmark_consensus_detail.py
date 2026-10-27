"""Measure the parameter count and latency of the frozen CDF pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, fuse_consensus_detail
from ssgf.models import Decoder, Encoder
from ssgf.models.baselines import IFCNN, fuse_l1
from ssgf.utils.checkpoint import load_ckpt


def count_parameters(*modules):
    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(r"D:\Code\_ssgf_external\m3fd_erf_pilot")
    parser.add_argument(
        "--visible", type=Path, default=root / "test_sources/visible/01400.png"
    )
    parser.add_argument(
        "--infrared",
        type=Path,
        default=root / "test_sources/infrared/01400.png",
    )
    parser.add_argument(
        "--mean", type=Path, default=root / "test_candidates/mean/01400.png"
    )
    parser.add_argument(
        "--ifcnn",
        type=Path,
        default=root / "test_candidates/ifcnn/01400.png",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            r"D:\Code\_server_handoff_v2\ssgf\results"
            r"\selective_fusion\cdf_complexity.json"
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    visible = np.asarray(Image.open(args.visible).convert("RGB"))
    infrared = np.asarray(Image.open(args.infrared).convert("L"))
    mean_image = np.asarray(Image.open(args.mean).convert("L"))
    ifcnn_image = np.asarray(Image.open(args.ifcnn).convert("L"))
    visible_t = (
        torch.from_numpy(visible.astype(np.float32) / 255.0)
        .mean(dim=2)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    infrared_t = (
        torch.from_numpy(infrared.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    config = get_default_config()
    encoder, decoder = Encoder(config), Decoder(config)
    load_ckpt("data/msrs_ae.pt", encoder=encoder, decoder=decoder)
    mean_rule = MeanRule(encoder, decoder, config, device)
    ifcnn = IFCNN().to(device).eval()
    ifcnn.load_state_dict(
        torch.load("data/btl/ifcnn.pt", map_location=device, weights_only=False)
    )

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    def mean_forward():
        return mean_rule.fuse_pair(visible_t, infrared_t)[0]

    def ifcnn_forward():
        encoded = fuse_l1(
            ifcnn.encode(visible_t * 2 - 1),
            ifcnn.encode(infrared_t * 2 - 1),
        )
        return (ifcnn.decode(encoded) + 1) / 2

    with torch.no_grad():
        for _ in range(args.warmup):
            mean_forward()
            ifcnn_forward()
            fuse_consensus_detail(
                visible,
                infrared,
                mean_image,
                ifcnn_image,
            )
        synchronize()

        mean_times, ifcnn_times, post_times, total_times = [], [], [], []
        for _ in range(args.runs):
            start = time.perf_counter()
            mean_output = mean_forward()
            synchronize()
            mean_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            ifcnn_output = ifcnn_forward()
            synchronize()
            ifcnn_times.append(time.perf_counter() - start)

            mean_np = (
                mean_output[0, 0].detach().cpu().numpy() * 255.0
            ).astype(np.uint8)
            ifcnn_np = (
                ifcnn_output[0, 0].detach().cpu().numpy() * 255.0
            ).astype(np.uint8)

            start = time.perf_counter()
            fuse_consensus_detail(
                visible,
                infrared,
                mean_np,
                ifcnn_np,
            )
            post_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            total_mean = mean_forward()
            total_ifcnn = ifcnn_forward()
            synchronize()
            total_mean_np = (
                total_mean[0, 0].detach().cpu().numpy() * 255.0
            ).astype(np.uint8)
            total_ifcnn_np = (
                total_ifcnn[0, 0].detach().cpu().numpy() * 255.0
            ).astype(np.uint8)
            fuse_consensus_detail(
                visible,
                infrared,
                total_mean_np,
                total_ifcnn_np,
            )
            synchronize()
            total_times.append(time.perf_counter() - start)

    payload = {
        "device": device,
        "resolution_hw": list(visible.shape[:2]),
        "runs": args.runs,
        "parameters": {
            "mean_autoencoder_frozen": count_parameters(encoder, decoder),
            "ifcnn_frozen": count_parameters(ifcnn),
            "cdf_trainable": 0,
        },
        "latency_ms": {
            name: {
                "mean": round(float(np.mean(values) * 1000), 3),
                "std": round(float(np.std(values) * 1000), 3),
            }
            for name, values in (
                ("mean_rule", mean_times),
                ("ifcnn", ifcnn_times),
                ("cdf_postprocess", post_times),
                ("end_to_end", total_times),
            )
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
