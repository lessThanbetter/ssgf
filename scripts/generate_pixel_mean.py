"""Generate standard pixel-mean fusion candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssgf.fusion import pixel_mean_fusion


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"{directory / stem}.*")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stems = sorted(
        {
            path.stem
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for path in (args.sources / "visible").glob(suffix)
        }
    )
    for stem in stems:
        visible = cv2.imread(
            str(find_image(args.sources / "visible", stem)),
            cv2.IMREAD_COLOR,
        )
        infrared = cv2.imread(
            str(find_image(args.sources / "infrared", stem)),
            cv2.IMREAD_GRAYSCALE,
        )
        cv2.imwrite(
            str(args.out / f"{stem}.png"),
            pixel_mean_fusion(visible, infrared),
        )
    print(f"WROTE {len(stems)} images to {args.out}")


if __name__ == "__main__":
    main()
