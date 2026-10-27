"""Benchmark the deterministic pixel-mean + IFCNN consensus-detail pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssgf.fusion import fuse_consensus_detail, pixel_mean_fusion
from ssgf.models.baselines import IFCNN, fuse_l1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/btl/ifcnn.pt",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = Path(r"D:\Code\_ssgf_external\m3fd_erf_pilot")
    visible_image = Image.open(
        root / "test_sources/visible/01400.png"
    ).convert("RGB")
    infrared_image = Image.open(
        root / "test_sources/infrared/01400.png"
    ).convert("L")
    if args.height and args.width:
        size = (args.width, args.height)
        visible_image = visible_image.resize(size, Image.BICUBIC)
        infrared_image = infrared_image.resize(size, Image.BICUBIC)
    visible = np.asarray(visible_image)
    infrared = np.asarray(infrared_image)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IFCNN().to(device).eval()
    model.load_state_dict(
        torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
    )
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

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(10):
            pixel_mean_fusion(visible, infrared)
            encoded = fuse_l1(
                model.encode(visible_t * 2 - 1),
                model.encode(infrared_t * 2 - 1),
            )
            output = ((model.decode(encoded) + 1) / 2)[0, 0]
            output = (output.cpu().numpy() * 255).astype(np.uint8)
            fuse_consensus_detail(
                visible,
                infrared,
                pixel_mean_fusion(visible, infrared),
                output,
            )
        synchronize()

        measurements = {
            "pixel_mean": [],
            "ifcnn": [],
            "cdf_postprocess": [],
            "end_to_end": [],
        }
        for _ in range(50):
            start = time.perf_counter()
            mean_image = pixel_mean_fusion(visible, infrared)
            measurements["pixel_mean"].append(time.perf_counter() - start)

            start = time.perf_counter()
            encoded = fuse_l1(
                model.encode(visible_t * 2 - 1),
                model.encode(infrared_t * 2 - 1),
            )
            fused_tensor = (model.decode(encoded) + 1) / 2
            synchronize()
            measurements["ifcnn"].append(time.perf_counter() - start)
            fused = (fused_tensor[0, 0].cpu().numpy() * 255).astype(np.uint8)

            start = time.perf_counter()
            fuse_consensus_detail(visible, infrared, mean_image, fused)
            measurements["cdf_postprocess"].append(
                time.perf_counter() - start
            )

            start = time.perf_counter()
            total_mean = pixel_mean_fusion(visible, infrared)
            encoded = fuse_l1(
                model.encode(visible_t * 2 - 1),
                model.encode(infrared_t * 2 - 1),
            )
            total_fused = (model.decode(encoded) + 1) / 2
            synchronize()
            total_fused = (
                total_fused[0, 0].cpu().numpy() * 255
            ).astype(np.uint8)
            fuse_consensus_detail(
                visible, infrared, total_mean, total_fused
            )
            measurements["end_to_end"].append(time.perf_counter() - start)

    payload = {
        "device": device,
        "resolution_hw": list(visible.shape[:2]),
        "runs": 50,
        "parameters": {
            "ifcnn_frozen": sum(p.numel() for p in model.parameters()),
            "cdf_trainable": 0,
            "pixel_mean_trainable": 0,
        },
        "latency_ms": {
            name: {
                "mean": round(float(np.mean(values) * 1000), 3),
                "std": round(float(np.std(values) * 1000), 3),
            }
            for name, values in measurements.items()
        },
    }
    output = args.output or Path(
        r"D:\Code\_server_handoff_v2\ssgf\results"
        r"\selective_fusion\cdf_pixelmean_complexity.json"
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
