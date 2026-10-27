"""Generate validity-aware regional fusion candidates on the validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CANDIDATES = (
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


def thermal_validity_mask(
    visible: np.ndarray,
    infrared: np.ndarray,
    threshold: float,
    visible_quantile: float,
) -> np.ndarray:
    visible_gray = cv2.cvtColor(visible, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    targetness = cv2.morphologyEx(infrared, cv2.MORPH_TOPHAT, kernel)
    cutoff = float(np.quantile(targetness, threshold))
    visible_cutoff = float(np.quantile(visible_gray, visible_quantile))
    raw = (
        (targetness >= cutoff)
        & (infrared >= 120)
        & (visible_gray < visible_cutoff)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/selective_fusion/validation_sources"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("results/selective_fusion/validation_candidates"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/selective_fusion/validation_candidates"),
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--visible-quantile", type=float, default=0.75)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--mode", choices=("blend", "detail"), default="blend")
    parser.add_argument("--detail-sigma", type=float, default=5.0)
    args = parser.parse_args()

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    stems = sorted(
        {
            path.stem
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for path in (args.sources / "visible").glob(suffix)
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"q{int(round(args.threshold * 100)):03d}_a{int(round(args.alpha * 100)):03d}"
    manifest = {
        "threshold": args.threshold,
        "visible_quantile": args.visible_quantile,
        "alpha": args.alpha,
        "mode": args.mode,
        "detail_sigma": args.detail_sigma,
        "methods": methods,
        "stems": len(stems),
    }
    (args.out / f"region_vasf_{tag}.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    for method in methods:
        prefix = "region" if args.mode == "blend" else "detail"
        suffix = tag
        if args.mode == "detail":
            suffix += f"_s{int(round(args.detail_sigma)):02d}"
        name = f"{prefix}_{method}_{suffix}"
        target = args.out / name
        target.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(stems, 1):
            visible = cv2.imread(
                str(find_image(args.sources / "visible", stem)),
                cv2.IMREAD_COLOR,
            )
            infrared = cv2.imread(
                str(find_image(args.sources / "infrared", stem)),
                cv2.IMREAD_GRAYSCALE,
            )
            fused = cv2.imread(
                str(find_image(args.candidates / method, stem)),
                cv2.IMREAD_GRAYSCALE,
            )
            if fused is None:
                raise FileNotFoundError(f"cannot read {method}/{stem}")
            if fused.shape != infrared.shape:
                fused = cv2.resize(
                    fused,
                    (infrared.shape[1], infrared.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            mask = thermal_validity_mask(
                visible, infrared, args.threshold, args.visible_quantile
            )
            if args.mode == "blend":
                fused_bgr = cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR).astype(np.float32)
                weight = args.alpha * mask[..., None]
                blended = (
                    visible.astype(np.float32) * (1.0 - weight) + fused_bgr * weight
                )
            else:
                lab = cv2.cvtColor(visible, cv2.COLOR_BGR2LAB).astype(np.float32)
                detail = fused.astype(np.float32) - cv2.GaussianBlur(
                    fused.astype(np.float32),
                    (0, 0),
                    sigmaX=args.detail_sigma,
                )
                lab[..., 0] += args.alpha * mask * detail
                blended = cv2.cvtColor(
                    np.clip(lab, 0.0, 255.0).astype(np.uint8),
                    cv2.COLOR_LAB2BGR,
                ).astype(np.float32)
            cv2.imwrite(
                str(target / f"{stem}.png"),
                np.clip(blended, 0.0, 255.0).astype(np.uint8),
            )
            if index % 50 == 0 or index == len(stems):
                print(f"{name}: {index}/{len(stems)}", flush=True)


if __name__ == "__main__":
    main()
