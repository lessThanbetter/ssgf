"""Benchmark YOLO11x-OBB inference on one DroneVehicle candidate image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, r"D:\Code\_server_handoff_v2\ssgf\scripts")
import torchvision_cuda_compat  # noqa: F401,E402

from ultralytics import YOLO  # noqa: E402


MODEL = Path(
    r"D:\Code\_ssgf_external\dronevehicle_val300\weights"
    r"\yolo11x-obb-best.pt"
)
IMAGE = Path(r"D:\Code\_ssgf_external\dronevehicle_test300\visible\test_00701.jpg")


def infer(model):
    return model.predict(
        str(IMAGE),
        imgsz=640,
        conf=0.001,
        iou=0.7,
        batch=1,
        verbose=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    model = YOLO(str(MODEL))
    for _ in range(3):
        infer(model)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(20):
        infer(model)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 20 * 1000
    peak = (torch.cuda.max_memory_allocated() - base) / (1024**2)
    result = {
        "model": "YOLO11x-OBB best",
        "device": torch.cuda.get_device_name(0),
        "input": 640,
        "batch": 1,
        "latency_ms": round(elapsed, 1),
        "extra_peak_mib": round(peak, 1),
    }
    print(
        f"YOLO11x-OBB input=640 batch=1 "
        f"latency_ms={result['latency_ms']:.1f} "
        f"extra_peak_MiB={result['extra_peak_mib']:.1f}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("WROTE", args.json)


if __name__ == "__main__":
    main()
