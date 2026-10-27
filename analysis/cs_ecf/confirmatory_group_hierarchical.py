"""Cluster-aware CS-ECF bootstrap using scenes and consecutive frame groups."""

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
pooled_indices = HELPERS["pooled_indices"]
choices_for = HELPERS["choices_for"]
DATASETS = HELPERS["DATASETS"]
DRAWS = 500


def group_ids(dataset, stems):
    if dataset == "llvip":
        return np.asarray([int(str(stem)[:3]) for stem in stems])
    if dataset == "m3fd":
        return np.asarray([int(str(stem)) // 100 for stem in stems])
    if dataset == "dronevehicle_test":
        return np.asarray(
            [
                (int(str(stem).split("_")[-1]) - 1) // 100
                for stem in stems
            ]
        )
    raise ValueError(dataset)


def pooled(rows, choices, indices):
    return pooled_indices(rows, choices, indices)


def main():
    rng = np.random.default_rng(2219)
    model_draws = {}
    report = {}
    for dataset, models in DATASETS.items():
        first_rows = model_rows(dataset, models[0])
        n_images = len(first_rows["vis"]["stems"])
        stems = np.asarray([str(value) for value in first_rows["vis"]["stems"]])
        groups = group_ids(dataset, stems)
        group_indices = {
            int(group): np.flatnonzero(groups == group)
            for group in sorted(set(groups))
        }
        sampled_groups = [
            rng.choice(
                sorted(group_indices),
                len(group_indices),
                replace=True,
            )
            for _ in range(DRAWS)
        ]
        dataset_model_draws = []
        model_report = {}
        point_gains = []
        for model in models:
            rows = model_rows(dataset, model)
            choices = choices_for(rows, n_images, 1.5)
            visible = ["vis"] * n_images
            all_indices = np.arange(n_images)
            point = pooled(rows, choices, all_indices) - pooled(
                rows, visible, all_indices
            )
            point_gains.append(point)
            draws = []
            per_group = {}
            for group, indices in group_indices.items():
                gain = pooled(rows, choices, indices) - pooled(
                    rows, visible, indices
                )
                per_group[str(group)] = round(float(gain), 4)
            for picked in sampled_groups:
                sample = np.concatenate(
                    [group_indices[int(group)] for group in picked]
                )
                draws.append(
                    pooled(rows, choices, sample)
                    - pooled(rows, visible, sample)
                )
            draws = np.asarray(draws)
            dataset_model_draws.append(draws)
            model_report[model] = {
                "gain": round(float(point), 4),
                "group_ci95": [
                    round(float(value), 4)
                    for value in np.percentile(draws, [2.5, 97.5])
                ],
                "per_group_gain": per_group,
            }
            print(dataset, model, model_report[model], flush=True)
        matrix = np.asarray(dataset_model_draws)
        model_draws[dataset] = matrix
        point_gains = np.asarray(point_gains)
        report[dataset] = {
            "n_groups": len(group_indices),
            "group_ids": sorted(group_indices),
            "mean_gain": round(float(point_gains.mean()), 4),
            "models": model_report,
            "fixed_models_group_ci95": [
                round(float(value), 4)
                for value in np.percentile(matrix.mean(axis=0), [2.5, 97.5])
            ],
        }

    hierarchical_draws = []
    for draw_index in range(DRAWS):
        dataset_values = []
        for dataset, models in DATASETS.items():
            sampled_models = rng.integers(0, len(models), len(models))
            dataset_values.append(
                model_draws[dataset][sampled_models, draw_index].mean()
            )
        hierarchical_draws.append(float(np.mean(dataset_values)))
    all_fixed = np.asarray(
        [model_draws[dataset].mean(axis=0) for dataset in DATASETS]
    )
    report["aggregate"] = {
        "datasets": list(DATASETS),
        "dataset_balanced_mean_gain": round(
            float(
                np.mean(
                    [
                        report[dataset]["mean_gain"]
                        for dataset in DATASETS
                    ]
                )
            ),
            4,
        ),
        "fixed_checkpoints_group_ci95": [
            round(float(value), 4)
            for value in np.percentile(all_fixed.mean(axis=0), [2.5, 97.5])
        ],
        "dataset_balanced_hierarchical_group_ci95": [
            round(float(value), 4)
            for value in np.percentile(hierarchical_draws, [2.5, 97.5])
        ],
        "note": (
            "M3FD contributes only two test groups and LLVIP seven scenes; "
            "cluster bootstrap resolution is correspondingly limited."
        ),
    }
    output = RESULT_ROOT / "confidence_shield_group_hierarchical.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("WROTE", output)


if __name__ == "__main__":
    main()
