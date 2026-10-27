"""Generate conservative visible-dominant blends for validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.50)
CANDIDATES = ("mean", "activity_only", "tophat", "probe", "densefuse")


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"{directory / stem}.*")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="data/selective_fusion/validation_sources")
    parser.add_argument(
        "--candidates", default="results/selective_fusion/validation_candidates"
    )
    parser.add_argument(
        "--out", default="results/selective_fusion/validation_candidates"
    )
    args = parser.parse_args()
    sources = Path(args.sources)
    candidates = Path(args.candidates)
    output = Path(args.out)
    stems = sorted(path.stem for path in (sources / "visible").glob("*.jpg"))

    for candidate in CANDIDATES:
        fused_dir = candidates / candidate
        for alpha in ALPHAS:
            name = f"blend_{candidate}_a{int(round(alpha * 100)):03d}"
            target = output / name
            target.mkdir(parents=True, exist_ok=True)
            for index, stem in enumerate(stems, 1):
                visible = cv2.imread(
                    str(find_image(sources / "visible", stem)),
                    cv2.IMREAD_COLOR,
                ).astype(np.float32)
                fused = cv2.imread(
                    str(find_image(fused_dir, stem)),
                    cv2.IMREAD_GRAYSCALE,
                )
                if fused is None:
                    raise RuntimeError(f"cannot read fused image for {candidate}/{stem}")
                fused = cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR).astype(np.float32)
                blended = np.clip((1.0 - alpha) * visible + alpha * fused, 0, 255)
                cv2.imwrite(str(target / f"{stem}.png"), blended.astype(np.uint8))
                if index % 50 == 0:
                    print(f"{name}: {index}/{len(stems)}", flush=True)


if __name__ == "__main__":
    main()
