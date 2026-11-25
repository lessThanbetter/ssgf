"""Generate PIAFusion candidates with batched inference."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE = SCRIPT_PATH.parents[1]
REPO_ROOT = next(
    path
    for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
    if (path / "data").is_dir()
)
EXTERNAL = Path(r"D:\Code\_ssgf_external")


def pairs(visible_dir: Path, infrared_dir: Path) -> list[tuple[Path, Path, str]]:
    visible = {path.stem: path for path in visible_dir.glob("*.jpg")}
    infrared = {path.stem: path for path in infrared_dir.glob("*.jpg")}
    return [
        (visible[stem], infrared[stem], stem)
        for stem in sorted(set(visible) & set(infrared))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        default=REPO_ROOT / "data" / "prospective_pia_sources",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "prospective_pia"
            / "candidates"
            / "piafusion"
        ),
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = EXTERNAL / "PIAFusion_pytorch"
    os.chdir(root)
    sys.path.insert(0, str(root))
    from models.common import RGB2YCrCb, clamp
    from models.fusion_model import PIAFusion

    model = PIAFusion().to(args.device).eval()
    model.load_state_dict(
        torch.load(
            root / "pretrained/fusion_model_epoch_29.pth",
            map_location="cpu",
            weights_only=False,
        )
    )
    items = pairs(args.sources / "visible", args.sources / "infrared")
    if not args.overwrite:
        items = [
            item
            for item in items
            if not (args.out / f"{item[2]}.png").is_file()
        ]
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"[pia-batch] {len(items)} remaining pairs, batch={args.batch}",
        flush=True,
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(items), args.batch):
            batch = items[offset : offset + args.batch]
            visible_tensors = []
            infrared_tensors = []
            for visible_path, infrared_path, _ in batch:
                visible_tensors.append(
                    TF.to_tensor(Image.open(visible_path).convert("RGB"))
                )
                infrared_tensors.append(
                    TF.to_tensor(Image.open(infrared_path).convert("L"))
                )
            visible = torch.stack(visible_tensors).to(args.device)
            infrared = torch.stack(infrared_tensors).to(args.device)
            luminance = (
                0.299 * visible[:, 0:1]
                + 0.587 * visible[:, 1:2]
                + 0.114 * visible[:, 2:3]
            ).clamp(0.0, 1.0)
            fused = clamp(model(luminance, infrared))[:, 0].clamp(0.0, 1.0)
            for (_, _, stem), image in zip(
                batch,
                fused.cpu().numpy(),
            ):
                cv2.imwrite(
                    str(args.out / f"{stem}.png"),
                    (image * 255.0).astype(np.uint8),
                )
            completed = min(offset + len(batch), len(items))
            if completed % 64 == 0 or completed == len(items):
                elapsed = time.perf_counter() - started
                print(
                    f"[pia-batch] {completed}/{len(items)} "
                    f"{completed / max(elapsed, 1e-9):.2f} images/s",
                    flush=True,
                )
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
