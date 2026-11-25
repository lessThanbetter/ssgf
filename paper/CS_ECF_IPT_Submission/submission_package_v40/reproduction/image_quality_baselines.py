"""Evaluate label-free image-statistics selectors against CS-ECF.

The selectors use only the registered visible/infrared pair and generated
candidates. They do not use detector confidence or ground truth. Image metrics
are computed on 256 x 256 grayscale inputs for a deterministic systems baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SSGF_DATA_ROOT", str(SCRIPT_ROOT)))
sys.path.insert(0, str(DATA_ROOT / "scripts"))

import analyze_csecf_vs_cdf as csecf  # noqa: E402
from local_ap_reference import load as local_load, pooled as local_pooled  # noqa: E402


VIF_HELPERS: dict = {}


def _read_image(path: Path, shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
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
    numerator = (2.0 * mu_ref * mu_candidate + c1) * (2.0 * covariance + c2)
    denominator = (
        (mu_ref * mu_ref + mu_candidate * mu_candidate + c1)
        * (var_ref + var_candidate + c2)
    )
    return float(np.mean(numerator / (denominator + 1e-12)))


def _quality_task(arguments):
    dataset, stem, visible_path, infrared_path, fused_paths = arguments
    global VIF_HELPERS
    if not VIF_HELPERS:
        VIF_HELPERS = runpy.run_path(
            str(DATA_ROOT / "scripts" / "vif_decomp2.py")
        )
    vif = VIF_HELPERS["vif"]
    visible = _read_image(Path(visible_path))
    infrared = _read_image(Path(infrared_path))
    result = {
        "dataset": dataset,
        "stem": stem,
        "entropy": {"vis": _entropy(visible)},
        "ssim": {},
        "vif": {},
    }
    for name, path in fused_paths.items():
        fused = _read_image(Path(path))
        result["entropy"][name] = _entropy(fused)
        result["ssim"][name] = 0.5 * (
            _ssim(visible, fused) + _ssim(infrared, fused)
        )
        result["vif"][name] = 0.5 * (
            vif(visible, fused, "canonical", True, 2.0 / (255.0 ** 2))
            + vif(infrared, fused, "canonical", True, 2.0 / (255.0 ** 2))
        )
    return result


def _dataset_paths(external_root: Path, data_root: Path) -> dict:
    cdf = "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"
    return {
        "llvip": {
            "visible": external_root / "r2_clean" / "test" / "visible",
            "infrared": external_root / "r2_clean" / "test" / "infrared",
            "fused": {
                "vis": data_root / "_deposit" / "ssgf-llvip-fusion-detection-v1" / "data" / "per_image" / "vis_split__vis.npz",
                "cdf": external_root / "r2_clean" / "test_candidates" / cdf,
                "pia": external_root / "r2_clean" / "sota_outs" / "pia_llvip_native_fp16",
                "spg": external_root / "r2_clean" / "sota_outs" / "spg_llvip_native_fp16",
            },
        },
        "m3fd": {
            "visible": external_root / "m3fd_erf_pilot" / "test_sources" / "visible",
            "infrared": external_root / "m3fd_erf_pilot" / "test_sources" / "infrared",
            "fused": {
                "cdf": external_root / "m3fd_erf_pilot" / "test_erf_candidates" / cdf,
                "pia": external_root / "m3fd_erf_pilot" / "test_sota_candidates" / "piafusion",
                "spg": external_root / "m3fd_erf_pilot" / "test_sota_candidates" / "spgfusion",
            },
        },
        "dronevehicle_test": {
            "visible": external_root / "dronevehicle_test300" / "visible",
            "infrared": external_root / "dronevehicle_test300" / "infrared",
            "fused": {
                "cdf": external_root / "dronevehicle_test300" / "erf_candidates" / cdf,
                "pia": external_root / "dronevehicle_test300" / "sota_candidates" / "piafusion",
                "spg": external_root / "dronevehicle_test300" / "sota_candidates" / "spgfusion",
            },
        },
    }


def _quality_choices(features: list[dict]) -> dict[str, list[str]]:
    fused = ("cdf", "pia", "spg")
    entropy_choices = []
    ssim_choices = []
    vif_choices = []
    for row in features:
        entropy_choices.append(
            max(("vis", *fused), key=lambda name: row["entropy"][name])
        )
        ssim_choices.append(
            max(fused, key=lambda name: row["ssim"][name])
        )
        vif_choices.append(
            max(fused, key=lambda name: row["vif"][name])
        )
    return {
        "max_entropy": entropy_choices,
        "max_ssim_fused": ssim_choices,
        "max_vif_fused": vif_choices,
    }


def _ap(stats: dict, indices: np.ndarray | None = None) -> float:
    if indices is None:
        indices = np.arange(stats["n"])
    return float(local_pooled(stats, np.asarray(indices, dtype=np.int64))[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features-out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    external_root = args.external_root.resolve()
    specs = csecf._candidate_paths()
    first_by_dataset = {}
    for spec in specs:
        first_by_dataset.setdefault(spec.dataset, spec)
    paths = _dataset_paths(external_root, data_root)
    tasks = []
    stems_by_dataset = {}
    for dataset, spec in first_by_dataset.items():
        states = csecf._load_checkpoint(spec)
        stems = [str(stem) for stem in states["vis"]["stems"]]
        stems_by_dataset[dataset] = stems
        dataset_paths = paths[dataset]
        for stem in stems:
            fused_paths = {
                name: str(_find(directory, stem))
                for name, directory in dataset_paths["fused"].items()
                if name != "vis"
            }
            tasks.append(
                (
                    dataset,
                    stem,
                    str(_find(dataset_paths["visible"], stem)),
                    str(_find(dataset_paths["infrared"], stem)),
                    fused_paths,
                )
            )

    if args.jobs == 1:
        features = [_quality_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            features = list(pool.map(_quality_task, tasks))
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    args.features_out.write_text(
        json.dumps(features, indent=2), encoding="utf-8"
    )

    by_dataset = {
        dataset: [row for row in features if row["dataset"] == dataset]
        for dataset in stems_by_dataset
    }
    choices = {
        dataset: _quality_choices(rows)
        for dataset, rows in by_dataset.items()
    }
    report = {"selectors": {}, "per_checkpoint": {}}
    for spec in specs:
        states = csecf._load_checkpoint(spec)
        baseline = _ap(states["vis"])
        dataset_choices = choices[spec.dataset]
        rows = {}
        for selector, selected in dataset_choices.items():
            stats = csecf._policy_stats(states, selected)
            rows[selector] = {
                "ap50": _ap(stats),
                "gain": _ap(stats) - baseline,
                "switch_fraction": (
                    sum(choice != "vis" for choice in selected) / len(selected)
                ),
                "selected_counts": {
                    candidate: selected.count(candidate)
                    for candidate in ("vis", "cdf", "pia", "spg")
                },
            }
        report["per_checkpoint"][f"{spec.dataset}/{spec.name}"] = rows

    for selector in ("max_entropy", "max_ssim_fused", "max_vif_fused"):
        dataset_means = {}
        for dataset in ("llvip", "m3fd", "dronevehicle_test"):
            values = [
                rows[selector]["gain"]
                for name, rows in report["per_checkpoint"].items()
                if name.startswith(dataset + "/")
            ]
            dataset_means[dataset] = float(np.mean(values))
        gains = [
            rows[selector]["gain"]
            for rows in report["per_checkpoint"].values()
        ]
        report["selectors"][selector] = {
            "dataset_balanced_gain": float(np.mean(list(dataset_means.values()))),
            "per_dataset_gain": dataset_means,
            "worst_checkpoint_gain": float(np.min(gains)),
            "positive_checkpoints": int(np.sum(np.asarray(gains) > 0)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["selectors"], indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
