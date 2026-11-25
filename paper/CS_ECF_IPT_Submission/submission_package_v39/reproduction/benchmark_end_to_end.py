"""Measure steady-state end-to-end CS-ECF pipeline latency on one GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE = SCRIPT_PATH.parents[1]
REPO_ROOT = next(
    path
    for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
    if (path / "scripts").is_dir() and (path / "ssgf").is_dir()
)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_PATH.parent))

import torchvision_cuda_compat  # noqa: E402,F401
from ultralytics import RTDETR, YOLO  # noqa: E402

from ssgf.fusion.consensus_detail import (  # noqa: E402
    DEFAULT_CONSENSUS_DETAIL,
    fuse_consensus_detail,
    pixel_mean_fusion,
)
from ssgf.models.baselines import IFCNN, fuse_l1  # noqa: E402


EXTERNAL = Path(r"D:\Code\_ssgf_external")
MODEL_SPECS = {
    "yolo26_s0": (
        "yolo",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\yolo26n_clean\weights\last.pt"
        ),
    ),
    "yolo26_s1": (
        "yolo",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\yolo26n_clean_s1\weights\last.pt"
        ),
    ),
    "yolo26_s2": (
        "yolo",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\yolo26n_clean_s2\weights\last.pt"
        ),
    ),
    "yolo8_s0": (
        "yolo",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\yolo8n_clean_holdout\weights\last.pt"
        ),
    ),
    "yolo8_s1": (
        "yolo",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\yolo8n_clean_holdout_s1\weights\last.pt"
        ),
    ),
    "rtdetr_l": (
        "rtdetr",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\rtdetr_l_clean\weights\last.pt"
        ),
    ),
}
CANDIDATES = ("vis", "cdf", "pia", "spg")
TOP_K = 5
MARGIN = 1.5


def install_mmseg_resize_shim() -> None:
    try:
        import mmseg.ops  # noqa: F401

        return
    except ImportError:
        pass

    def resize(
        input,
        size=None,
        scale_factor=None,
        mode="bilinear",
        align_corners=None,
        **kwargs,
    ):
        del kwargs
        return F.interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )

    mmseg = types.ModuleType("mmseg")
    ops = types.ModuleType("mmseg.ops")
    ops.resize = resize
    mmseg.ops = ops
    sys.modules["mmseg"] = mmseg
    sys.modules["mmseg.ops"] = ops


def topk_score(result) -> float:
    conf = result.boxes.conf.detach().cpu().numpy()
    if len(conf) == 0:
        return -1.0
    return float(np.sort(conf)[-TOP_K:].mean())


class PipelineModels:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        pia_root = EXTERNAL / "PIAFusion_pytorch"
        sys.path.insert(0, str(pia_root))
        from models.common import RGB2YCrCb, clamp
        from models.fusion_model import PIAFusion

        self.pia = PIAFusion().to(device).eval()
        self.pia.load_state_dict(
            torch.load(
                pia_root / "pretrained/fusion_model_epoch_29.pth",
                map_location="cpu",
                weights_only=False,
            )
        )
        self.rgb_to_ycbcr = RGB2YCrCb
        self.clamp = clamp

        spg_root = EXTERNAL / "SPGFusion"
        install_mmseg_resize_shim()
        sys.path.insert(0, str(spg_root))
        from model.Dino_Clip import DINOiser
        from model.SPGFusion import SPGFusion

        clip_config = {
            "clip_model": "ViT-B/16",
            "clip_model_pth": str(
                spg_root / "model/ViT-B-16-laion2B/open_clip_pytorch_model.bin"
            ),
            "patch_size": 16,
            "img_size": 224,
        }
        self.extractor = DINOiser(clip_config).to(device).eval()
        self.extractor.load_dino()
        self.extractor.to(device).eval()
        self.spg = SPGFusion().to(device).eval()
        checkpoint = torch.load(
            spg_root / "CHECKPOINT/best_model_1.pth",
            map_location="cpu",
            weights_only=False,
        )
        state = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        self.spg.load_state_dict(state, strict=True)

        self.ifcnn = IFCNN().to(device).eval()
        self.ifcnn.load_state_dict(
            torch.load(
                REPO_ROOT / "data/btl/ifcnn.pt",
                map_location=device,
                weights_only=False,
            )
        )

    def pia_fusion(
        self,
        visible_rgb: np.ndarray,
        infrared: np.ndarray,
    ) -> np.ndarray:
        visible = torch.from_numpy(
            visible_rgb.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        infrared_t = torch.from_numpy(
            infrared.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            luminance = (
                0.299 * visible[:, 0:1]
                + 0.587 * visible[:, 1:2]
                + 0.114 * visible[:, 2:3]
            ).clamp(0.0, 1.0)
            fused = self.clamp(
                self.pia(luminance, infrared_t)
            )[0, 0].clamp(0.0, 1.0)
        return (fused.cpu().numpy() * 255.0).astype(np.uint8)

    def spg_fusion(
        self,
        visible_rgb: np.ndarray,
        infrared: np.ndarray,
        max_side: int = 512,
    ) -> np.ndarray:
        visible = torch.from_numpy(
            visible_rgb.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        infrared_rgb = torch.from_numpy(
            np.repeat(infrared[..., None], 3, axis=2).astype(np.float32)
            / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)
        output_size = visible.shape[-2:]
        if max_side and max(output_size) > max_side:
            scale = max_side / max(output_size)
            input_size = (
                max(16, round(output_size[0] * scale / 16) * 16),
                max(16, round(output_size[1] * scale / 16) * 16),
            )
            visible = F.interpolate(
                visible, size=input_size, mode="bilinear", align_corners=False
            )
            infrared_rgb = F.interpolate(
                infrared_rgb, size=input_size, mode="bilinear", align_corners=False
            )
        with torch.inference_mode():
            visible_semantic = self.extractor(visible)
            infrared_semantic = self.extractor(infrared_rgb)
            fused = self.spg(
                visible,
                infrared_rgb,
                visible_semantic,
                infrared_semantic,
            )[0]
            gray = (
                0.299 * fused[0] + 0.587 * fused[1] + 0.114 * fused[2]
            ).float().clamp(0.0, 1.0)
            if gray.shape[-2:] != output_size:
                gray = F.interpolate(
                    gray[None, None],
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
        return (gray.cpu().numpy() * 255.0).astype(np.uint8)

    def cdf_fusion(
        self,
        visible_bgr: np.ndarray,
        infrared: np.ndarray,
    ) -> np.ndarray:
        mean_candidate = pixel_mean_fusion(visible_bgr, infrared)
        visible_gray = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2GRAY)
        visible_t = torch.from_numpy(
            visible_gray.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(self.device)
        infrared_t = torch.from_numpy(
            infrared.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            encoded = fuse_l1(
                self.ifcnn.encode(visible_t * 2 - 1),
                self.ifcnn.encode(infrared_t * 2 - 1),
            )
            ifcnn_output = (self.ifcnn.decode(encoded) + 1) / 2
        ifcnn_candidate = (
            ifcnn_output[0, 0].clamp(0.0, 1.0).cpu().numpy() * 255.0
        ).astype(np.uint8)
        output, _, _ = fuse_consensus_detail(
            visible_bgr,
            infrared,
            mean_candidate,
            ifcnn_candidate,
            DEFAULT_CONSENSUS_DETAIL,
        )
        return output


def predict(model, image: np.ndarray, device: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = model.predict(
        image,
        conf=0.001,
        iou=0.7,
        imgsz=640,
        max_det=300,
        device=device,
        verbose=False,
    )[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def select(results: dict[str, object]) -> tuple[str, object]:
    scores = {
        name: topk_score(result)
        for name, result in results.items()
    }
    best = max(results, key=lambda name: scores[name])
    selected = (
        best
        if best != "vis" and scores[best] >= MARGIN * scores["vis"]
        else "vis"
    )
    return selected, results[selected]


def run_trial(
    style: str,
    detector,
    fusion: PipelineModels,
    visible_rgb: np.ndarray,
    visible_bgr: np.ndarray,
    infrared: np.ndarray,
    device: str,
) -> dict:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    fusion_started = time.perf_counter()
    pia = fusion.pia_fusion(visible_rgb, infrared) if style != "visible" else None
    spg = (
        fusion.spg_fusion(visible_rgb, infrared)
        if style == "full"
        else None
    )
    cdf = (
        fusion.cdf_fusion(visible_bgr, infrared)
        if style == "full"
        else None
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fusion_ms = (time.perf_counter() - fusion_started) * 1000.0

    detector_started = time.perf_counter()
    visible_result, _ = predict(detector, visible_rgb, device)
    results = {"vis": visible_result}
    if style in ("budgeted", "full"):
        pia_result, _ = predict(detector, pia, device)
        results["pia"] = pia_result
    if style == "full":
        cdf_result, _ = predict(detector, cdf, device)
        spg_result, _ = predict(detector, spg, device)
        results["cdf"] = cdf_result
        results["spg"] = spg_result
    detector_ms = (time.perf_counter() - detector_started) * 1000.0
    selected, _ = select(results)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "total_ms": (time.perf_counter() - started) * 1000.0,
        "fusion_ms": fusion_ms,
        "detector_ms": detector_ms,
        "selected": selected,
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            REPO_ROOT
            / "data"
            / "prospective_pia_sources"
        ),
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            PACKAGE
            / "evidence"
            / "prospective_pia_confirmation"
            / "end_to_end_latency.json"
        ),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    device = f"cuda:{args.device}"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    visible_paths = sorted((args.source / "visible").glob("*.jpg"))[: args.frames]
    if len(visible_paths) != args.frames:
        raise RuntimeError("not enough benchmark frames")
    inputs = []
    for path in visible_paths:
        infrared_path = args.source / "infrared" / path.name
        visible_rgb = cv2.cvtColor(
            cv2.imread(str(path), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        visible_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        infrared = cv2.imread(str(infrared_path), cv2.IMREAD_GRAYSCALE)
        inputs.append((path.stem, visible_rgb, visible_bgr, infrared))

    fusion = PipelineModels(device)
    results = {}
    for model_name in args.models:
        kind, path = MODEL_SPECS[model_name]
        detector = RTDETR(str(path)) if kind == "rtdetr" else YOLO(str(path))
        for style in ("visible", "budgeted", "full"):
            for index in range(args.warmup):
                _, visible_rgb, visible_bgr, infrared = inputs[
                    index % len(inputs)
                ]
                run_trial(
                    style,
                    detector,
                    fusion,
                    visible_rgb,
                    visible_bgr,
                    infrared,
                    args.device,
                )
            trials = []
            for index in range(args.runs):
                stem, visible_rgb, visible_bgr, infrared = inputs[
                    index % len(inputs)
                ]
                trial = run_trial(
                    style,
                    detector,
                    fusion,
                    visible_rgb,
                    visible_bgr,
                    infrared,
                    args.device,
                )
                trial["stem"] = stem
                trials.append(trial)
            totals = [row["total_ms"] for row in trials]
            fusions = [row["fusion_ms"] for row in trials]
            detectors = [row["detector_ms"] for row in trials]
            results[f"{model_name}/{style}"] = {
                "median_total_ms": percentile(totals, 50),
                "p95_total_ms": percentile(totals, 95),
                "median_fusion_ms": percentile(fusions, 50),
                "median_detector_ms": percentile(detectors, 50),
                "selected_pia_fraction": float(np.mean([
                    row["selected"] == "pia" for row in trials
                ])),
                "trials": trials,
            }
            print(
                f"{model_name:10s} {style:8s} "
                f"median={percentile(totals, 50):.1f}ms "
                f"p95={percentile(totals, 95):.1f}ms",
                flush=True,
            )
        del detector
        torch.cuda.empty_cache()

    payload = {
        "device": torch.cuda.get_device_name(0),
        "scope": (
            "steady-state single-process pipeline after images are decoded; "
            "models already loaded; batch size 1; CUDA synchronized per trial"
        ),
        "warmup": args.warmup,
        "runs": args.runs,
        "frames": [row[0] for row in inputs],
        "styles": {
            "visible": "one visible-input detector forward",
            "budgeted": "PIAFusion + visible/PIA detector forwards + selection",
            "full": (
                "PIAFusion + SPGFusion + CDF + four detector forwards + selection"
            ),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
