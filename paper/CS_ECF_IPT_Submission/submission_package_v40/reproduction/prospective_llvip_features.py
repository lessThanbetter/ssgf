"""Compute image-only scores for the prospective LLVIP sequence audit."""

from __future__ import annotations

import argparse
import json
import runpy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi


CDF_NAME = "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"
VIF_HELPERS: dict = {}


def _read_image(path: Path, shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape != shape:
        image = cv2.resize(
            image,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return image.astype(np.float32) / 255.0


def _find(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(directory / stem)


def _entropy(image: np.ndarray) -> float:
    histogram = np.histogram(image, bins=64, range=(0.0, 1.0))[0]
    probability = histogram.astype(np.float64)
    probability /= probability.sum() + 1e-12
    return float(-(probability * np.log2(probability + 1e-12)).sum())


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    sigma = 1.5
    mu_ref = ndi.gaussian_filter(reference, sigma)
    mu_candidate = ndi.gaussian_filter(candidate, sigma)
    ref_sq = ndi.gaussian_filter(reference * reference, sigma)
    candidate_sq = ndi.gaussian_filter(candidate * candidate, sigma)
    cross = ndi.gaussian_filter(reference * candidate, sigma)
    var_ref = ref_sq - mu_ref * mu_ref
    var_candidate = candidate_sq - mu_candidate * mu_candidate
    covariance = cross - mu_ref * mu_candidate
    numerator = (2.0 * mu_ref * mu_candidate + c1) * (
        2.0 * covariance + c2
    )
    denominator = (
        (mu_ref * mu_ref + mu_candidate * mu_candidate + c1)
        * (var_ref + var_candidate + c2)
    )
    return float(np.mean(numerator / (denominator + 1e-12)))


def _quality_task(arguments):
    sequence, stem, visible_path, infrared_path, candidate_paths, vif_path = (
        arguments
    )
    global VIF_HELPERS
    if not VIF_HELPERS:
        VIF_HELPERS = runpy.run_path(str(vif_path))
    vif = VIF_HELPERS["vif"]
    visible = _read_image(Path(visible_path))
    infrared = _read_image(Path(infrared_path))
    candidates = {"vis": visible}
    candidates.update({
        name: _read_image(Path(path))
        for name, path in candidate_paths.items()
    })
    result = {
        "sequence": sequence,
        "stem": stem,
        "entropy": {},
        "ssim": {},
        "vif": {},
    }
    for name, candidate in candidates.items():
        result["entropy"][name] = _entropy(candidate)
        result["ssim"][name] = 0.5 * (
            _ssim(visible, candidate) + _ssim(infrared, candidate)
        )
        result["vif"][name] = 0.5 * (
            vif(
                visible,
                candidate,
                "canonical",
                True,
                2.0 / (255.0 ** 2),
            )
            + vif(
                infrared,
                candidate,
                "canonical",
                True,
                2.0 / (255.0 ** 2),
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vif-script", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    sequence = args.source.name.removeprefix("erf_holdout_llvip")
    visible_dir = args.source / "visible"
    infrared_dir = args.source / "infrared"
    stems = sorted(path.stem for path in visible_dir.glob("*.jpg"))
    tasks = []
    for stem in stems:
        candidate_paths = {
            "cdf": _find(args.candidates / CDF_NAME, stem),
            "pia": _find(args.candidates / "piafusion", stem),
            "spg": _find(args.candidates / "spgfusion", stem),
        }
        tasks.append((
            sequence,
            stem,
            _find(visible_dir, stem),
            _find(infrared_dir, stem),
            candidate_paths,
            args.vif_script.resolve(),
        ))

    if args.jobs == 1:
        rows = [_quality_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(_quality_task, tasks))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"WROTE {args.out} rows={len(rows)}")


if __name__ == "__main__":
    main()
