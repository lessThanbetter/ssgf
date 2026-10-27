"""Compare fixed fusion inputs and CS-ECF across all three test datasets."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULT_ROOT = ROOT / "results/selective_fusion"
HELPERS = runpy.run_path(
    r"D:\Code\_server_handoff_v2\tmp\confirmatory_three_dataset_summary.py"
)
model_rows = HELPERS["model_rows"]
pooled = HELPERS["pooled"]
choices_for = HELPERS["choices_for"]
DATASETS = HELPERS["DATASETS"]
CANDIDATES = ("vis", "cdf", "pia", "spg")


def main():
    three_dataset = json.loads(
        (RESULT_ROOT / "confidence_shield_three_dataset_summary.json").read_text(
            encoding="utf-8"
        )
    )
    rows_by_model = {}
    values = {
        policy: {}
        for policy in (
            "always_cdf",
            "always_pia",
            "always_spg",
            "no_shield",
            "cs_ecf",
        )
    }
    for dataset, models in DATASETS.items():
        for model in models:
            rows = model_rows(dataset, model)
            n_images = len(rows["vis"]["stems"])
            visible = pooled(rows, ["vis"] * n_images)
            fixed = {
                candidate: pooled(rows, [candidate] * n_images)
                - visible
                for candidate in CANDIDATES
            }
            no_shield = pooled(
                rows, choices_for(rows, n_images, 1.0)
            ) - visible
            cs_ecf = three_dataset[dataset]["models"][model]["gain"]
            rows_by_model[(dataset, model)] = {
                "visible_ap50": round(visible, 4),
                "always_cdf": round(fixed["cdf"], 4),
                "always_pia": round(fixed["pia"], 4),
                "always_spg": round(fixed["spg"], 4),
                "no_shield": round(no_shield, 4),
                "cs_ecf": cs_ecf,
            }
            for policy, key in (
                ("always_cdf", "always_cdf"),
                ("always_pia", "always_pia"),
                ("always_spg", "always_spg"),
                ("no_shield", "no_shield"),
                ("cs_ecf", "cs_ecf"),
            ):
                values[policy][(dataset, model)] = (
                    rows_by_model[(dataset, model)][key]
                )

    summaries = {}
    for policy, policy_values in values.items():
        profile_means = {
            dataset: round(
                float(
                    np.mean(
                        [
                            policy_values[(dataset, model)]
                            for model in models
                        ]
                    )
                ),
                4,
            )
            for dataset, models in DATASETS.items()
        }
        gains = np.asarray(list(policy_values.values()))
        summaries[policy] = {
            "positive_checkpoints": int(np.sum(gains > 0)),
            "mean_gain": round(float(gains.mean()), 4),
            "min_gain": round(float(gains.min()), 4),
            "max_gain": round(float(gains.max()), 4),
            "profile_means": profile_means,
            "dataset_balanced_mean": round(
                float(np.mean(list(profile_means.values()))), 4
            ),
        }

    output = {
        "datasets": list(DATASETS),
        "checkpoints": 12,
        "per_checkpoint": {
            f"{dataset}/{model}": result
            for (dataset, model), result in rows_by_model.items()
        },
        "summaries": summaries,
    }
    output_path = RESULT_ROOT / "confirmatory_policy_comparison.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print("WROTE", output_path)


if __name__ == "__main__":
    main()
