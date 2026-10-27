"""Three-dataset hierarchical bootstrap for the frozen CS-ECF policy."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULT_ROOT = ROOT / "results/selective_fusion"
HELPERS = runpy.run_path(
    r"D:\Code\_server_handoff_v2\tmp\confidence_shield_summary.py"
)
rows_for = HELPERS["rows_for"]
pooled = HELPERS["pooled"]
pooled_indices = HELPERS["pooled_indices"]
choices_for = HELPERS["choices_for"]
DRAWS = 400
MARGIN = 1.5
DATASETS = {
    "llvip": [
        "yolo11",
        "agnostic",
        "yolo26_s0",
        "rtdetr",
        "yolo26_s1",
        "yolo26_s2",
        "yolo8_s0",
        "yolo8_s1",
    ],
    "m3fd": ["yolo26_m3fd6", "yolo8n_m3fd6"],
    "dronevehicle_test": ["best", "last"],
}
DRONE_FILES = {
    "vis": "vis",
    "cdf": "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05",
    "pia": "piafusion",
    "spg": "spgfusion",
}


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def drone_rows(model):
    if model == "best":
        directory = RESULT_ROOT / "dronevehicle_test_scores"
        prefix = "yolo11x_obb"
    elif model == "last":
        directory = RESULT_ROOT / "dronevehicle_test_scores_last"
        prefix = "yolo11x_obb_last"
    else:
        raise ValueError(model)
    return {
        candidate: load_npz(
            directory / f"{prefix}__{filename}.npz"
        )
        for candidate, filename in DRONE_FILES.items()
    }


def model_rows(dataset, model):
    if dataset == "dronevehicle_test":
        return drone_rows(model)
    return rows_for(dataset, model)


def bootstrap_mean_ci(values, rng):
    values = np.asarray(values, dtype=np.float64)
    draws = [
        rng.choice(values, len(values), replace=True).mean()
        for _ in range(DRAWS)
    ]
    return [
        round(float(value), 4)
        for value in np.percentile(draws, [2.5, 97.5])
    ]


def main():
    rng = np.random.default_rng(7309)
    report = {}
    image_matrices = {}
    for dataset, models in DATASETS.items():
        model_image_draws = []
        point_gains = []
        model_report = {}
        first_rows = model_rows(dataset, models[0])
        n_images = len(first_rows["vis"]["stems"])
        bootstrap_indices = [
            rng.integers(0, n_images, n_images) for _ in range(DRAWS)
        ]
        for model in models:
            rows = model_rows(dataset, model)
            choices = choices_for(rows, n_images, MARGIN)
            baseline = pooled(rows, ["vis"] * n_images)
            selected = pooled(rows, choices)
            gain = selected - baseline
            point_gains.append(gain)
            draws = []
            for indices in bootstrap_indices:
                visible_draw = pooled_indices(
                    rows, ["vis"] * n_images, indices
                )
                draws.append(
                    pooled_indices(rows, choices, indices) - visible_draw
                )
            draws = np.asarray(draws)
            model_image_draws.append(draws)
            model_report[model] = {
                "visible_ap50": round(baseline, 4),
                "selected_ap50": round(selected, 4),
                "gain": round(gain, 4),
                "image_ci95": [
                    round(float(value), 4)
                    for value in np.percentile(draws, [2.5, 97.5])
                ],
                "selected_counts": {
                    candidate: choices.count(candidate)
                    for candidate in rows
                },
            }
            print(dataset, model, model_report[model], flush=True)
        matrix = np.asarray(model_image_draws)
        image_matrices[dataset] = matrix
        gains = np.asarray(point_gains)
        report[dataset] = {
            "models": model_report,
            "summary": {
                "n_models": len(models),
                "positive_models": int(np.sum(gains > 0)),
                "mean_gain": round(float(gains.mean()), 4),
                "min_gain": round(float(gains.min()), 4),
                "max_gain": round(float(gains.max()), 4),
                "model_ci95": bootstrap_mean_ci(gains, rng),
                "image_ci95": [
                    round(float(value), 4)
                    for value in np.percentile(
                        matrix.mean(axis=0), [2.5, 97.5]
                    )
                ],
            },
        }
        print(dataset, "summary", report[dataset]["summary"], flush=True)

    dataset_point_means = np.asarray(
        [report[dataset]["summary"]["mean_gain"] for dataset in DATASETS]
    )
    fixed_dataset_draws = np.asarray(
        [image_matrices[dataset].mean(axis=0) for dataset in DATASETS]
    )
    all_model_draws = np.concatenate(
        [image_matrices[dataset] for dataset in DATASETS], axis=0
    )
    hierarchical_draws = []
    model_only_draws = []
    for draw_index in range(DRAWS):
        dataset_draws = []
        dataset_model_only = []
        for dataset, models in DATASETS.items():
            sampled_models = rng.integers(0, len(models), len(models))
            dataset_draws.append(
                image_matrices[dataset][sampled_models, draw_index].mean()
            )
            point_gains = np.asarray(
                [
                    report[dataset]["models"][model]["gain"]
                    for model in models
                ]
            )
            dataset_model_only.append(
                point_gains[sampled_models].mean()
            )
        hierarchical_draws.append(np.mean(dataset_draws))
        model_only_draws.append(np.mean(dataset_model_only))

    report["aggregate"] = {
        "datasets": list(DATASETS),
        "n_checkpoints": int(sum(len(models) for models in DATASETS.values())),
        "positive_checkpoints": int(
            sum(
                report[dataset]["summary"]["positive_models"]
                for dataset in DATASETS
            )
        ),
        "dataset_balanced_mean": round(
            float(dataset_point_means.mean()), 4
        ),
        "min_checkpoint_gain": round(
            float(
                min(
                    report[dataset]["summary"]["min_gain"]
                    for dataset in DATASETS
                )
            ),
            4,
        ),
        "max_checkpoint_gain": round(
            float(
                max(
                    report[dataset]["summary"]["max_gain"]
                    for dataset in DATASETS
                )
            ),
            4,
        ),
        "fixed_checkpoints_image_ci95": [
            round(float(value), 4)
            for value in np.percentile(
                fixed_dataset_draws.mean(axis=0), [2.5, 97.5]
            )
        ],
        "all_checkpoints_image_ci95": [
            round(float(value), 4)
            for value in np.percentile(
                all_model_draws.mean(axis=0), [2.5, 97.5]
            )
        ],
        "dataset_balanced_model_ci95": [
            round(float(value), 4)
            for value in np.percentile(model_only_draws, [2.5, 97.5])
        ],
        "dataset_balanced_hierarchical_ci95": [
            round(float(value), 4)
            for value in np.percentile(hierarchical_draws, [2.5, 97.5])
        ],
    }
    output = RESULT_ROOT / "confidence_shield_three_dataset_summary.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print("WROTE", output)


if __name__ == "__main__":
    main()
