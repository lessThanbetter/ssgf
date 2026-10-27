"""Compare thermal-mask and global CDF injection on DroneVehicle test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Code\_server_handoff_v2\ssgf\scripts")
from local_ap_reference import ap_per_class  # noqa: E402


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULT_ROOT = ROOT / "results/selective_fusion"
MODELS = {
    "best": (
        RESULT_ROOT / "dronevehicle_test_scores",
        "yolo11x_obb",
    ),
    "last": (
        RESULT_ROOT / "dronevehicle_test_scores_last",
        "yolo11x_obb_last",
    ),
}


def load(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def ap50(data):
    return float(
        ap_per_class(
            data["tp"],
            data["conf"],
            data["pcls"],
            data["tcls"],
        )[0]
    )


def main():
    report = {}
    for model, (directory, prefix) in MODELS.items():
        visible = ap50(load(directory / f"{prefix}__vis.npz"))
        masked = ap50(
            load(
                directory
                / f"{prefix}__consensus_intersection_pixel_mean_ifcnn_q095_a020_s05.npz"
            )
        )
        global_injection = ap50(
            load(
                directory
                / f"{prefix}__consensus_intersection_pixel_mean_ifcnn_global.npz"
            )
        )
        report[model] = {
            "visible_ap50": round(visible, 4),
            "thermal_mask_ap50": round(masked, 4),
            "global_injection_ap50": round(global_injection, 4),
            "thermal_mask_gain": round(masked - visible, 4),
            "global_gain": round(global_injection - visible, 4),
            "mask_minus_global": round(masked - global_injection, 4),
        }
    report["mean"] = {
        "thermal_mask_gain": round(
            float(
                np.mean(
                    [
                        report[model]["thermal_mask_gain"]
                        for model in MODELS
                    ]
                )
            ),
            4,
        ),
        "global_gain": round(
            float(
                np.mean(
                    [report[model]["global_gain"] for model in MODELS]
                )
            ),
            4,
        ),
        "mask_minus_global": round(
            float(
                np.mean(
                    [
                        report[model]["mask_minus_global"]
                        for model in MODELS
                    ]
                )
            ),
            4,
        ),
    }
    output = RESULT_ROOT / "ipt_mask_ablation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("WROTE", output)


if __name__ == "__main__":
    main()
