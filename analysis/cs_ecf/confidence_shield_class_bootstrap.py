"""Bootstrap per-class effects of the frozen confidence-shield policy."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULTS = ROOT / "results/selective_fusion"
CLASS_HELPERS = runpy.run_path(
    r"D:\Code\_server_handoff_v2\tmp\confidence_shield_class_diagnostics.py"
)
rows_for = CLASS_HELPERS["rows_for"]
class_ap50 = CLASS_HELPERS["class_ap50"]
choices_for = CLASS_HELPERS["choices_for"]
PROFILES = CLASS_HELPERS["PROFILES"]
CLASS_NAMES = CLASS_HELPERS["CLASS_NAMES"]
DRAWS = 400


def main():
    rng = np.random.default_rng(9173)
    report = {}
    for profile, models in PROFILES.items():
        report[profile] = {}
        first_rows = rows_for(profile, models[0])
        n_images = len(first_rows["vis"]["stems"])
        bootstrap_indices = [
            rng.integers(0, n_images, n_images) for _ in range(DRAWS)
        ]
        class_ids = sorted(CLASS_NAMES[profile])
        for model in models:
            rows = rows_for(profile, model)
            choices = choices_for(rows, n_images, 1.5)
            visible = ["vis"] * n_images
            report[profile][model] = {}
            for class_id in class_ids:
                point_visible = class_ap50(rows, visible, [class_id])[
                    str(class_id)
                ]
                point_shield = class_ap50(rows, choices, [class_id])[
                    str(class_id)
                ]
                draws = []
                for indices in bootstrap_indices:
                    visible_ap = class_ap50(
                        rows, visible, [class_id], indices
                    )[str(class_id)]
                    shield_ap = class_ap50(
                        rows, choices, [class_id], indices
                    )[str(class_id)]
                    draws.append(shield_ap - visible_ap)
                ci = np.percentile(draws, [2.5, 97.5])
                result = {
                    "class_name": CLASS_NAMES[profile][class_id],
                    "visible_ap50": round(point_visible, 4),
                    "shield_ap50": round(point_shield, 4),
                    "gain": round(point_shield - point_visible, 4),
                    "image_ci95": [
                        round(float(value), 4) for value in ci
                    ],
                }
                report[profile][model][str(class_id)] = result
                print(profile, model, class_id, result, flush=True)
    output_path = RESULTS / "confidence_shield_class_bootstrap.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", output_path)


if __name__ == "__main__":
    main()
