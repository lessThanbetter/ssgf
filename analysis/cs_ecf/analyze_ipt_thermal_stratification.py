"""Analyze whether CS-ECF switching follows thermal and visible evidence."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import fisher_exact, spearmanr


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
sys.path.insert(0, str(ROOT))
from ssgf.fusion import thermal_validity_mask  # noqa: E402

RESULT_ROOT = ROOT / "results/selective_fusion"
HELPERS = runpy.run_path(
    r"D:\Code\_server_handoff_v2\tmp\confirmatory_three_dataset_summary.py"
)
model_rows = HELPERS["model_rows"]
pooled_indices = HELPERS["pooled_indices"]
choices_for = HELPERS["choices_for"]
DATASETS = HELPERS["DATASETS"]
SOURCES = {
    "llvip": Path(r"D:\Code\_ssgf_external\llvip_sources"),
    "m3fd": Path(r"D:\Code\_ssgf_external\m3fd_erf_pilot\test_sources"),
    "dronevehicle_test": Path(r"D:\Code\_ssgf_external\dronevehicle_test300"),
}


def find_image(directory, stem):
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"{directory / stem}.*")


def pooled(rows, choices, indices):
    return pooled_indices(rows, choices, indices)


def image_features(dataset, visible_path, infrared_path):
    visible = cv2.imread(str(visible_path), cv2.IMREAD_COLOR)
    infrared = cv2.imread(str(infrared_path), cv2.IMREAD_GRAYSCALE)
    if dataset == "dronevehicle_test":
        # Remove the 100-pixel white border used by DroneVehicle.
        border = 100
        visible = visible[border:-border, border:-border]
        infrared = infrared[border:-border, border:-border]
    gray = cv2.cvtColor(visible, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    top_hat = cv2.morphologyEx(infrared, cv2.MORPH_TOPHAT, kernel)
    mask = thermal_validity_mask(visible, infrared)
    return {
        "visible_luma": float(gray.mean() / 255.0),
        "ir_mean": float(infrared.mean() / 255.0),
        "thermal_tophat_mean": float(top_hat.mean() / 255.0),
        "thermal_tophat_q95": float(np.quantile(top_hat, 0.95) / 255.0),
        "thermal_mask_mean": float(mask.mean()),
        "thermal_mask_active_fraction": float(np.mean(mask > 0.2)),
    }


def main():
    report = {}
    for dataset, models in DATASETS.items():
        rows = {model: model_rows(dataset, model) for model in models}
        stems = [str(stem) for stem in rows[models[0]]["vis"]["stems"]]
        source = SOURCES[dataset]
        features = [
            image_features(
                dataset,
                find_image(source / "visible", stem),
                find_image(source / "infrared", stem),
            )
            for stem in stems
        ]
        thermal = np.asarray(
            [row["thermal_tophat_q95"] for row in features]
        )
        visible = np.asarray([row["visible_luma"] for row in features])
        mask_active = np.asarray(
            [row["thermal_mask_active_fraction"] for row in features]
        )
        thermal_threshold = float(np.median(thermal))
        visible_threshold = float(np.median(visible))
        high_thermal_low_visible = (
            (thermal >= thermal_threshold) & (visible <= visible_threshold)
        )
        rest = ~high_thermal_low_visible

        model_reports = {}
        switch_fraction = np.zeros(len(stems), dtype=np.float64)
        for model in models:
            model_data = rows[model]
            n_images = len(stems)
            choices = choices_for(model_data, n_images, 1.5)
            selected = np.asarray(
                [choice != "vis" for choice in choices], dtype=np.float64
            )
            switch_fraction += selected / len(models)
            all_indices = np.arange(n_images)
            regime_report = {}
            for regime_name, regime in (
                ("high_thermal_low_visible", high_thermal_low_visible),
                ("rest", rest),
            ):
                indices = np.flatnonzero(regime)
                selected_ap = pooled(model_data, choices, indices)
                visible_ap = pooled(
                    model_data, ["vis"] * n_images, indices
                )
                regime_report[regime_name] = {
                    "n_images": int(len(indices)),
                    "switch_rate": round(float(selected[indices].mean()), 4),
                    "visible_ap50": round(visible_ap, 4),
                    "selected_ap50": round(selected_ap, 4),
                    "gain": round(selected_ap - visible_ap, 4),
                }
            hot = selected[high_thermal_low_visible]
            other = selected[rest]
            table = [
                [int(hot.sum()), int(len(hot) - hot.sum())],
                [int(other.sum()), int(len(other) - other.sum())],
            ]
            odds_ratio, p_value = fisher_exact(table)
            model_reports[model] = {
                **regime_report,
                "switch_rate_difference": round(
                    float(hot.mean() - other.mean()), 4
                ),
                "fisher_odds_ratio": round(float(odds_ratio), 4),
                "fisher_p": round(float(p_value), 6),
            }

        thermal_rho, thermal_p = spearmanr(thermal, switch_fraction)
        visible_rho, visible_p = spearmanr(visible, switch_fraction)
        mask_rho, mask_p = spearmanr(mask_active, switch_fraction)
        report[dataset] = {
            "n_images": len(stems),
            "feature_thresholds": {
                "thermal_tophat_q95_median": round(thermal_threshold, 6),
                "visible_luma_median": round(visible_threshold, 6),
            },
            "correlation_with_switch_fraction": {
                "thermal_rho": round(float(thermal_rho), 4),
                "thermal_p": round(float(thermal_p), 6),
                "visible_rho": round(float(visible_rho), 4),
                "visible_p": round(float(visible_p), 6),
                "thermal_mask_rho": round(float(mask_rho), 4),
                "thermal_mask_p": round(float(mask_p), 6),
            },
            "models": model_reports,
        }
        print(dataset, json.dumps(report[dataset], indent=2), flush=True)

    output = RESULT_ROOT / "ipt_thermal_stratification.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", output)


if __name__ == "__main__":
    main()
