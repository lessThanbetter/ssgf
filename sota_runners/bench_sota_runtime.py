"""Latency, parameter count and extra peak GPU memory for PIAFusion/SPGFusion."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def run(fn, iters: int, warm: int = 3) -> float:
    for _ in range(warm):
        result = fn()
        del result
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        result = fn()
        del result
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def peak_mb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del result
    return peak / (1024 ** 2)


def time_pia(root: Path, device: str) -> dict:
    sys.path.insert(0, str(root))
    from models.fusion_model import PIAFusion

    model = PIAFusion().to(device).eval()
    model.load_state_dict(torch.load(
        root / "pretrained/fusion_model_epoch_29.pth",
        map_location="cpu", weights_only=False,
    ))
    vis = torch.rand(1, 1, 384, 512, device=device)
    ir = torch.rand(1, 1, 384, 512, device=device)

    def infer():
        with torch.inference_mode():
            return model(vis, ir)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    result = {
        "params_m": round(params, 2),
        "latency_ms": round(run(infer, 20), 1),
        "extra_peak_mib": round(peak_mb(infer), 1),
    }
    print(f"PIAFusion {params:10.2f} {result['latency_ms']:10.1f} "
          f"{result['extra_peak_mib']:16.1f}")
    return result


def time_spg(root: Path, device: str) -> dict:
    try:
        import mmseg.ops  # noqa: F401
    except ImportError:
        mmseg = types.ModuleType("mmseg")
        ops = types.ModuleType("mmseg.ops")

        def resize(input, size=None, scale_factor=None, mode="bilinear",
                   align_corners=None, **kwargs):
            del kwargs
            return F.interpolate(input, size=size, scale_factor=scale_factor,
                                 mode=mode, align_corners=align_corners)

        ops.resize = resize
        mmseg.ops = ops
        sys.modules["mmseg"] = mmseg
        sys.modules["mmseg.ops"] = ops

    sys.path.insert(0, str(root))
    from model.Dino_Clip import DINOiser
    from model.SPGFusion import SPGFusion

    clip_cfg = {
        "clip_model": "ViT-B/16",
        "clip_model_pth": str(
            root / "model/ViT-B-16-laion2B/open_clip_pytorch_model.bin"
        ),
        "patch_size": 16,
        "img_size": 224,
    }
    extractor = DINOiser(clip_cfg).to(device).eval()
    extractor.load_dino()
    extractor.to(device).eval()

    fusion = SPGFusion().to(device).eval()
    checkpoint = torch.load(
        root / "CHECKPOINT/best_model_1.pth",
        map_location="cpu", weights_only=False,
    )
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    fusion.load_state_dict(state, strict=True)

    vis = torch.rand(1, 3, 384, 512, device=device)
    ir = torch.rand(1, 3, 384, 512, device=device)

    def infer():
        with torch.inference_mode():
            vi_sem = extractor(vis)
            ir_sem = extractor(ir)
            return fusion(vis, ir, vi_sem, ir_sem)

    params = sum(p.numel() for p in fusion.parameters()) / 1e6
    result = {
        "params_m": round(params, 2),
        "latency_ms": round(run(infer, 5, warm=2), 1),
        "extra_peak_mib": round(peak_mb(infer), 1),
    }
    print(f"SPGFusion {params:10.2f} {result['latency_ms']:10.1f} "
          f"{result['extra_peak_mib']:16.1f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pia-root", default="sota/PIAFusion")
    parser.add_argument("--spg-root", default="sota/SPGFusion")
    parser.add_argument("--pia", action="store_true")
    parser.add_argument("--spg", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the memory benchmark")
    print(f"device={torch.cuda.get_device_name(0)} input=384x512 FP32 batch=1")
    print(f"{'method':10s} {'params(M)':>10s} {'time(ms)':>10s} {'extra_peak(MiB)':>16s}")
    results = {}
    if args.pia or not args.spg:
        results["piafusion"] = time_pia(
            Path(args.pia_root).resolve(), args.device
        )
    if args.spg or not args.pia:
        results["spgfusion"] = time_spg(
            Path(args.spg_root).resolve(), args.device
        )
    if args.json:
        payload = {
            "device": torch.cuda.get_device_name(0),
            "input_height": 384,
            "input_width": 512,
            "batch": 1,
            "dtype": "float32",
            "methods": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("WROTE", args.json)


if __name__ == "__main__":
    main()
