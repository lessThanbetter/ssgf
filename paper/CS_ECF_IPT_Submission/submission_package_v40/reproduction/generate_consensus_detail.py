"""Generate cross-method consensus thermal-detail injections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


METHODS = (
    "mean",
    "tophat",
    "probe",
    "ifcnn",
    "piafusion",
    "spgfusion",
)


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"{directory / stem}.*")


def thermal_mask(visible, infrared, threshold=0.95, visible_quantile=0.75):
    visible_gray = cv2.cvtColor(visible, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    targetness = cv2.morphologyEx(infrared, cv2.MORPH_TOPHAT, kernel)
    raw = (
        (targetness >= np.quantile(targetness, threshold))
        & (infrared >= 120)
        & (visible_gray < np.quantile(visible_gray, visible_quantile))
    ).astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(raw, 8)
    cleaned = np.zeros_like(raw)
    for label in range(1, count):
        if statistics[label, cv2.CC_STAT_AREA] >= 64:
            cleaned[labels == label] = 255
    cleaned = cv2.dilate(
        cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    cleaned = cv2.GaussianBlur(cleaned, (31, 31), 0)
    return cleaned.astype(np.float32) / 255.0


def highpass(image, sigma):
    image = image.astype(np.float32)
    return image - cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--min-magnitude", type=float, default=0.0)
    parser.add_argument(
        "--mask-mode",
        choices=("thermal", "global"),
        default="thermal",
        help="Use the thermal-validity mask or inject detail over the full image.",
    )
    parser.add_argument(
        "--policy-names",
        default="consensus_mean_ifcnn,consensus_mean_probe_ifcnn_spg,consensus_median6,consensus_trimmed6",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stems = sorted(
        {
            path.stem
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for path in (args.sources / "visible").glob(suffix)
        }
    )
    all_policies = {
        "consensus_mean_ifcnn": ("mean", "ifcnn"),
        "consensus_pixel_mean_ifcnn": ("pixel_mean", "ifcnn"),
        "consensus_mean_probe_ifcnn_spg": (
            "mean",
            "probe",
            "ifcnn",
            "spgfusion",
        ),
        "consensus_median6": METHODS,
        "consensus_trimmed6": METHODS,
        "consensus_intersection_mean_ifcnn": ("mean", "ifcnn"),
        "consensus_intersection_pixel_mean_ifcnn": ("pixel_mean", "ifcnn"),
        "consensus_sign_median6": METHODS,
        "consensus_intersection_budget004": ("mean", "ifcnn"),
        "consensus_intersection_budget005": ("mean", "ifcnn"),
        "consensus_intersection_budget006": ("mean", "ifcnn"),
        "consensus_intersection_cap006": ("mean", "ifcnn"),
        "consensus_intersection_cap007": ("mean", "ifcnn"),
        "consensus_intersection_cap008": ("mean", "ifcnn"),
        "consensus_intersection_gate009": ("mean", "ifcnn"),
        "consensus_intersection_gate010": ("mean", "ifcnn"),
    }
    requested = [
        name.strip() for name in args.policy_names.split(",") if name.strip()
    ]
    policies = {name: all_policies[name] for name in requested}
    required_methods = sorted(
        {method for members in policies.values() for method in members}
    )
    tag = (
        f"q{int(round(args.threshold * 100)):03d}_"
        f"a{int(round(args.alpha * 100)):03d}_"
        f"s{int(round(args.sigma)):02d}"
    )
    if args.min_magnitude:
        tag += f"_m{int(round(args.min_magnitude * 10)):02d}"
    target_dirs = {
        name: args.out / f"{name}_{tag}" for name in policies
    }
    for directory in target_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "threshold": args.threshold,
        "alpha": args.alpha,
        "sigma": args.sigma,
        "tag": tag,
        "min_magnitude": args.min_magnitude,
        "mask_mode": args.mask_mode,
        "methods": METHODS,
        "policies": {
            name: list(members) for name, members in policies.items()
        },
    }
    (args.out / "consensus_detail_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    for index, stem in enumerate(stems, 1):
        visible = cv2.imread(
            str(find_image(args.sources / "visible", stem)),
            cv2.IMREAD_COLOR,
        )
        infrared = cv2.imread(
            str(find_image(args.sources / "infrared", stem)),
            cv2.IMREAD_GRAYSCALE,
        )
        mask = (
            thermal_mask(visible, infrared, args.threshold)
            if args.mask_mode == "thermal"
            else np.ones(infrared.shape, dtype=np.float32)
        )
        details = {}
        for method in required_methods:
            fused = cv2.imread(
                str(find_image(args.candidates / method, stem)),
                cv2.IMREAD_GRAYSCALE,
            )
            if fused.shape != infrared.shape:
                fused = cv2.resize(
                    fused,
                    (infrared.shape[1], infrared.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            details[method] = highpass(fused, args.sigma)

        stack = np.stack([details[method] for method in required_methods])
        sorted_stack = np.sort(stack, axis=0)
        aggregate = {}
        for name in policies:
            if name == "consensus_median6":
                aggregate[name] = np.median(stack, axis=0)
            elif name == "consensus_trimmed6":
                aggregate[name] = np.mean(sorted_stack[1:-1], axis=0)
            elif (
                name == "consensus_intersection_mean_ifcnn"
                or name == "consensus_intersection_pixel_mean_ifcnn"
                or name.startswith("consensus_intersection_budget")
                or name.startswith("consensus_intersection_cap")
                or name.startswith("consensus_intersection_gate")
            ):
                first_name = (
                    "pixel_mean" if "pixel_mean" in name else "mean"
                )
                first = details[first_name]
                second = details["ifcnn"]
                aggregate[name] = np.where(
                    (first * second > 0)
                    & (
                        np.minimum(np.abs(first), np.abs(second))
                        >= args.min_magnitude
                    ),
                    np.sign(first + second) * np.minimum(
                        np.abs(first), np.abs(second)
                    ),
                    0.0,
                )
            elif name == "consensus_sign_median6":
                positive = np.sum(stack > 0, axis=0)
                negative = np.sum(stack < 0, axis=0)
                vote = (positive >= 4) | (negative >= 4)
                aggregate[name] = np.where(vote, np.median(stack, axis=0), 0.0)
            else:
                aggregate[name] = np.mean(
                    [details[method] for method in policies[name]],
                    axis=0,
                )
        for name, detail in aggregate.items():
            effective_alpha = args.alpha
            if name.startswith("consensus_intersection_budget"):
                budget = int(name[-3:]) / 1000.0
                active_fraction = float(np.mean(mask > 0.20))
                effective_alpha *= min(
                    1.0, budget / (active_fraction + 1e-8)
                )
            elif name.startswith("consensus_intersection_cap"):
                cap = int(name[-3:]) / 1000.0
                active_fraction = float(np.mean(mask > 0.20))
                effective_alpha *= min(
                    1.0, cap / (active_fraction + 1e-8)
                )
            elif name.startswith("consensus_intersection_gate"):
                cap = int(name[-3:]) / 1000.0
                active_fraction = float(np.mean(mask > 0.20))
                if active_fraction > cap:
                    effective_alpha = 0.0
            lab = cv2.cvtColor(visible, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[..., 0] += effective_alpha * mask * detail
            output = cv2.cvtColor(
                np.clip(lab, 0.0, 255.0).astype(np.uint8),
                cv2.COLOR_LAB2BGR,
            )
            cv2.imwrite(str(target_dirs[name] / f"{stem}.png"), output)
        if index % 50 == 0 or index == len(stems):
            print(f"{index}/{len(stems)}", flush=True)


if __name__ == "__main__":
    main()
