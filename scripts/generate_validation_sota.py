"""Generate PIAFusion and SPGFusion inputs for the clean validation split."""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F


EXTERNAL = Path(r"D:\Code\_ssgf_external")


def install_mmseg_resize_shim() -> None:
    """Provide the only MMSeg operation used by SPGFusion without installing MMSeg."""

    def resize(
        input,
        size=None,
        scale_factor=None,
        mode="bilinear",
        align_corners=None,
        **kwargs,
    ):
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
    sys.modules.setdefault("mmseg", mmseg)
    sys.modules.setdefault("mmseg.ops", ops)


def pairs(visible_dir: Path, infrared_dir: Path):
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif")
    visible = {
        path.stem: path
        for extension in extensions
        for path in visible_dir.glob(f"*{extension}")
    }
    infrared = {
        path.stem: path
        for extension in extensions
        for path in infrared_dir.glob(f"*{extension}")
    }
    return [
        (visible[stem], infrared[stem], stem)
        for stem in sorted(set(visible) & set(infrared))
    ]


def native_tensors(visible_path: Path, infrared_path: Path):
    visible = Image.open(visible_path).convert("RGB")
    infrared = Image.open(infrared_path).convert("L")
    return TF.to_tensor(visible), TF.to_tensor(infrared)


def run_pia(
    visible_dir: Path,
    infrared_dir: Path,
    output_dir: Path,
    checkpoint: Path,
) -> None:
    root = EXTERNAL / "PIAFusion_pytorch"
    os.chdir(root)
    sys.path.insert(0, str(root))
    from models.common import RGB2YCrCb, clamp
    from models.fusion_model import PIAFusion

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PIAFusion().to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = pairs(visible_dir, infrared_dir)
    print(f"[pia] {len(items)} pairs -> {output_dir}", flush=True)
    with torch.no_grad():
        for index, (visible_path, infrared_path, stem) in enumerate(items, 1):
            visible, infrared = native_tensors(visible_path, infrared_path)
            visible, infrared = visible.to(device), infrared.to(device)
            luminance, _, _ = RGB2YCrCb(visible)
            fused = clamp(model(luminance.unsqueeze(0), infrared.unsqueeze(0)))[0, 0]
            fused = fused.clamp(0.0, 1.0).cpu().numpy()
            cv2.imwrite(
                str(output_dir / f"{stem}.png"),
                (fused * 255.0).astype(np.uint8),
            )
            if index % 25 == 0 or index == len(items):
                print(f"[pia] {index}/{len(items)}", flush=True)


def run_spg(
    visible_dir: Path,
    infrared_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    half: bool,
    max_side: int,
    limit: int,
) -> None:
    root = EXTERNAL / "SPGFusion"
    os.chdir(root)
    sys.path.insert(0, str(root))
    install_mmseg_resize_shim()
    from model.Dino_Clip import DINOiser
    from model.SPGFusion import SPGFusion

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_config = {
        "clip_model": "ViT-B/16",
        "clip_model_pth": "model/ViT-B-16-laion2B/open_clip_pytorch_model.bin",
        "patch_size": 16,
        "img_size": 224,
    }
    extractor = DINOiser(clip_config).to(device)
    extractor.load_dino()
    extractor.to(device)
    extractor.eval()
    model = SPGFusion().to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state, strict=True)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = pairs(visible_dir, infrared_dir)
    if limit:
        items = items[:limit]
    print(f"[spg] {len(items)} pairs -> {output_dir} (half={half})", flush=True)
    context = (
        torch.autocast("cuda", dtype=torch.float16)
        if half
        else torch.autocast("cuda", enabled=False)
    )
    with torch.no_grad(), context:
        for index, (visible_path, infrared_path, stem) in enumerate(items, 1):
            visible = TF.to_tensor(Image.open(visible_path).convert("RGB")).unsqueeze(0)
            infrared = TF.to_tensor(Image.open(infrared_path).convert("RGB")).unsqueeze(0)
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
                infrared = F.interpolate(
                    infrared, size=input_size, mode="bilinear", align_corners=False
                )
            visible, infrared = visible.to(device), infrared.to(device)
            if half:
                visible, infrared = visible.half(), infrared.half()
            visible_semantic = extractor(visible)
            infrared_semantic = extractor(infrared)
            fused = model(visible, infrared, visible_semantic, infrared_semantic)[0]
            gray = (
                0.299 * fused[0] + 0.587 * fused[1] + 0.114 * fused[2]
            ).float().clamp(0.0, 1.0)
            if gray.shape[-2:] != output_size:
                gray = F.interpolate(
                    gray.unsqueeze(0).unsqueeze(0),
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            cv2.imwrite(
                str(output_dir / f"{stem}.png"),
                (gray.cpu().numpy() * 255.0).astype(np.uint8),
            )
            if index % 10 == 0 or index == len(items):
                print(f"[spg] {index}/{len(items)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/selective_fusion/validation_sources"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/selective_fusion/validation_candidates"),
    )
    parser.add_argument("--method", choices=("pia", "spg", "both"), default="both")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--spg-max-side", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    visible_dir = args.sources.resolve() / "visible"
    infrared_dir = args.sources.resolve() / "infrared"
    output_dir = args.out.resolve()
    if args.method in ("pia", "both"):
        run_pia(
            visible_dir,
            infrared_dir,
            output_dir / "piafusion",
            EXTERNAL / "PIAFusion_pytorch/pretrained/fusion_model_epoch_29.pth",
        )
    if args.method in ("spg", "both"):
        run_spg(
            visible_dir,
            infrared_dir,
            output_dir / "spgfusion",
            EXTERNAL / "SPGFusion/CHECKPOINT/best_model_1.pth",
            args.half,
            args.spg_max_side,
            args.limit,
        )


if __name__ == "__main__":
    main()
