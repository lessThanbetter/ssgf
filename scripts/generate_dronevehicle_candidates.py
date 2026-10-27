"""Generate pixel-mean and IFCNN candidates for DroneVehicle test pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssgf.fusion import pixel_mean_fusion
from ssgf.models.baselines import IFCNN, fuse_l1


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"{directory / stem}.*")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ifcnn-checkpoint",
        type=Path,
        default=Path("data/btl/ifcnn.pt"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ifcnn = IFCNN().to(device).eval()
    ifcnn.load_state_dict(
        torch.load(
            args.ifcnn_checkpoint,
            map_location=device,
            weights_only=False,
        )
    )
    stems = sorted(
        {
            path.stem
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for path in (args.sources / "visible").glob(suffix)
        }
    )
    if args.limit:
        stems = stems[: args.limit]
    pixel_mean_directory = args.out / "pixel_mean"
    ifcnn_directory = args.out / "ifcnn"
    pixel_mean_directory.mkdir(parents=True, exist_ok=True)
    ifcnn_directory.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index, stem in enumerate(stems, 1):
            visible = cv2.imread(
                str(find_image(args.sources / "visible", stem)),
                cv2.IMREAD_COLOR,
            )
            infrared = cv2.imread(
                str(find_image(args.sources / "infrared", stem)),
                cv2.IMREAD_GRAYSCALE,
            )
            cv2.imwrite(
                str(pixel_mean_directory / f"{stem}.png"),
                pixel_mean_fusion(visible, infrared),
            )
            visible_gray = cv2.cvtColor(
                visible, cv2.COLOR_BGR2GRAY
            ).astype(np.float32) / 255.0
            infrared_float = infrared.astype(np.float32) / 255.0
            visible_tensor = (
                torch.from_numpy(visible_gray)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            infrared_tensor = (
                torch.from_numpy(infrared_float)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            encoded = fuse_l1(
                ifcnn.encode(visible_tensor * 2 - 1),
                ifcnn.encode(infrared_tensor * 2 - 1),
            )
            fused = (ifcnn.decode(encoded)[0, 0] + 1) / 2
            fused = fused.clamp(0, 1).cpu().numpy()
            cv2.imwrite(
                str(ifcnn_directory / f"{stem}.png"),
                (fused * 255).astype(np.uint8),
            )
            if index % 25 == 0 or index == len(stems):
                print(f"{index}/{len(stems)}", flush=True)


if __name__ == "__main__":
    main()
