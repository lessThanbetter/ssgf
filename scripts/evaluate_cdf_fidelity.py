"""Compare CDF image-fusion fidelity with its base candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssgf.utils.metrics import report


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"{directory / stem}.*")


def grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return image.astype(np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    stems = manifest["stems"]["test"][: args.limit]
    methods = {
        "pixel_mean": root / "test_candidates/pixel_mean",
        "ifcnn": root / "test_candidates/ifcnn",
        "cdf": root
        / "test_erf_candidates"
        / "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05",
    }
    rows = {method: [] for method in methods}
    for index, stem in enumerate(stems, 1):
        visible = grayscale(find_image(root / "test_sources/visible", stem))
        infrared = grayscale(find_image(root / "test_sources/infrared", stem))
        for method, directory in methods.items():
            fused = grayscale(find_image(directory, stem))
            rows[method].append(report(fused, visible, infrared))
        if index % 10 == 0:
            print(f"{index}/{len(stems)}", flush=True)
    summary = {}
    for method, values in rows.items():
        summary[method] = {
            metric: {
                "mean": round(
                    float(np.mean([row[metric] for row in values])), 6
                ),
                "std": round(
                    float(np.std([row[metric] for row in values])), 6
                ),
            }
            for metric in values[0]
        }
    payload = {
        "dataset": "M3FD",
        "split": "test",
        "images": len(stems),
        "methods": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
