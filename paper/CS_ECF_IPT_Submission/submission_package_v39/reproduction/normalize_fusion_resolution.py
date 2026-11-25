"""Upsample resized fusion outputs to the registered source resolution."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(directory / stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("piafusion", "spgfusion"),
    )
    args = parser.parse_args()

    visible_dir = args.sources / "visible"
    stems = sorted(path.stem for path in visible_dir.glob("*.jpg"))
    for method in args.methods:
        source_dir = args.candidates / method
        backup_dir = args.candidates / f"{method}_512"
        if source_dir.exists() and not backup_dir.exists():
            source_dir.rename(backup_dir)
        source_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            source = find_image(backup_dir, stem)
            visible = cv2.imread(
                str(find_image(visible_dir, stem)),
                cv2.IMREAD_COLOR,
            )
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(source)
            resized = cv2.resize(
                image,
                (visible.shape[1], visible.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imwrite(str(source_dir / f"{stem}.png"), resized)
        print(
            f"normalized {method}: {len(stems)} images; "
            f"original-size outputs remain available in {backup_dir.name}"
        )


if __name__ == "__main__":
    main()
