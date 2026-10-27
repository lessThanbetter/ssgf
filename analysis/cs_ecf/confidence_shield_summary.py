"""Bootstrap the validation-selected confidence-shield policy."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Code\_server_handoff_v2\ssgf\scripts")
from local_ap_reference import ap_per_class  # noqa: E402


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULTS = ROOT / "results/selective_fusion"
HELPERS = runpy.run_path(
    r"D:\Code\_server_handoff_v2\tmp\confidence_candidate_selection.py"
)
rows_for = HELPERS["rows_for"]
score_confidence = HELPERS["confidence_score"]
DRAWS = 400
TOP_K = 5
MARGIN = 1.5
SELECTION_PATH = RESULTS / "confidence_candidate_selection.json"
PROFILES = {
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
}


def bootstrap_mean_ci(values, rng):
    values = np.asarray(values, dtype=np.float64)
    draws = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(DRAWS)
    ]
    return [
        round(float(value), 4)
        for value in np.percentile(draws, [2.5, 97.5])
    ]


def pooled(rows, choices):
    tp, conf, pcls, tcls = [], [], [], []
    n_gt = rows["vis"]["n_gt"]
    for image_index, candidate in enumerate(choices):
        data = rows[candidate]
        mask = data["img_idx"] == image_index
        tp.append(data["tp"][mask])
        conf.append(data["conf"][mask])
        pcls.append(data["pcls"][mask])
        start = int(n_gt[:image_index].sum())
        stop = int(n_gt[: image_index + 1].sum())
        tcls.append(data["tcls"][start:stop])
    return float(
        ap_per_class(
            np.concatenate(tp),
            np.concatenate(conf),
            np.concatenate(pcls),
            np.concatenate(tcls),
        )[0]
    )


def pooled_indices(rows, choices, indices):
    tp, conf, pcls, tcls = [], [], [], []
    n_gt = rows["vis"]["n_gt"]
    for out_index, image_index in enumerate(indices):
        data = rows[choices[image_index]]
        mask = data["img_idx"] == image_index
        tp.append(data["tp"][mask])
        conf.append(data["conf"][mask])
        pcls.append(data["pcls"][mask])
        start = int(n_gt[:image_index].sum())
        stop = int(n_gt[: image_index + 1].sum())
        tcls.append(data["tcls"][start:stop])
    return float(
        ap_per_class(
            np.concatenate(tp),
            np.concatenate(conf),
            np.concatenate(pcls),
            np.concatenate(tcls),
        )[0]
    )


def choices_for(rows, n_images, margin):
    choices = []
    for image_index in range(n_images):
        values = {
            candidate: score_confidence(data, image_index, "top5")
            for candidate, data in rows.items()
        }
        selected = max(values, key=values.get)
        if selected != "vis" and values[selected] < margin * values["vis"]:
            selected = "vis"
        choices.append(selected)
    return choices


def main():
    rng = np.random.default_rng(2231)
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    report = {}
    profile_draws = {}
    all_gains = []
    all_no_shield_gains = []
    all_shield_deltas = []
    for profile, models in PROFILES.items():
        report[profile] = {}
        model_draws = []
        no_shield_model_draws = []
        shield_delta_model_draws = []
        first_rows = rows_for(profile, models[0])
        n_images = len(first_rows["vis"]["stems"])
        bootstrap_indices = [
            rng.integers(0, n_images, n_images) for _ in range(DRAWS)
        ]
        for model in models:
            rows = rows_for(profile, model)
            if len(rows["vis"]["stems"]) != n_images:
                raise ValueError(
                    f"Image counts differ within profile {profile}: "
                    f"{n_images} vs {len(rows['vis']['stems'])}"
                )
            choices = choices_for(rows, n_images, MARGIN)
            no_shield_choices = choices_for(rows, n_images, 1.0)
            baseline = pooled(rows, ["vis"] * n_images)
            selected = pooled(rows, choices)
            no_shield_selected = pooled(rows, no_shield_choices)
            gain = selected - baseline
            no_shield_gain = no_shield_selected - baseline
            all_gains.append(gain)
            all_no_shield_gains.append(no_shield_gain)
            all_shield_deltas.append(gain - no_shield_gain)
            image_draws = []
            no_shield_image_draws = []
            shield_delta_draws = []
            for indices in bootstrap_indices:
                baseline_draw = pooled_indices(
                    rows, ["vis"] * n_images, indices
                )
                shield_draw = (
                    pooled_indices(rows, choices, indices) - baseline_draw
                )
                no_shield_draw = (
                    pooled_indices(rows, no_shield_choices, indices)
                    - baseline_draw
                )
                image_draws.append(shield_draw)
                no_shield_image_draws.append(no_shield_draw)
                shield_delta_draws.append(shield_draw - no_shield_draw)
            image_draws = np.asarray(image_draws, dtype=np.float64)
            no_shield_image_draws = np.asarray(
                no_shield_image_draws, dtype=np.float64
            )
            shield_delta_draws = np.asarray(
                shield_delta_draws, dtype=np.float64
            )
            model_draws.append(image_draws)
            no_shield_model_draws.append(no_shield_image_draws)
            shield_delta_model_draws.append(shield_delta_draws)
            recorded_no_shield_gain = selection[profile][model]["top5"]["gain"]
            if abs(no_shield_gain - recorded_no_shield_gain) > 0.00011:
                raise ValueError(
                    f"No-shield replay mismatch for {profile}/{model}: "
                    f"{no_shield_gain} vs {recorded_no_shield_gain}"
                )
            report[profile][model] = {
                "visible_ap50": round(baseline, 4),
                "selected_ap50": round(selected, 4),
                "gain": round(gain, 4),
                "image_ci95": [
                    round(float(value), 4)
                    for value in np.percentile(image_draws, [2.5, 97.5])
                ],
                "no_shield_gain": round(no_shield_gain, 4),
                "no_shield_image_ci95": [
                    round(float(value), 4)
                    for value in np.percentile(
                        no_shield_image_draws, [2.5, 97.5]
                    )
                ],
                "shield_increment_vs_no_shield": round(
                    gain - no_shield_gain, 4
                ),
                "shield_delta_image_ci95": [
                    round(float(value), 4)
                    for value in np.percentile(
                        shield_delta_draws, [2.5, 97.5]
                    )
                ],
                "counts": {
                    candidate: choices.count(candidate)
                    for candidate in rows
                },
            }
            print(profile, model, report[profile][model], flush=True)
        model_draws = np.asarray(model_draws, dtype=np.float64)
        no_shield_model_draws = np.asarray(
            no_shield_model_draws, dtype=np.float64
        )
        shield_delta_model_draws = np.asarray(
            shield_delta_model_draws, dtype=np.float64
        )
        profile_draws[profile] = model_draws
        point_gains = np.asarray(
            [
                report[profile][model]["selected_ap50"]
                - report[profile][model]["visible_ap50"]
                for model in models
            ],
            dtype=np.float64,
        )
        no_shield_point_gains = np.asarray(
            [
                report[profile][model]["no_shield_gain"]
                for model in models
            ],
            dtype=np.float64,
        )
        shield_delta_point_gains = np.asarray(
            [
                report[profile][model]["shield_increment_vs_no_shield"]
                for model in models
            ],
            dtype=np.float64,
        )
        model_indices = [
            rng.integers(0, len(models), len(models)) for _ in range(DRAWS)
        ]
        hierarchical_draws = np.asarray(
            [
                model_draws[indices, draw_index].mean()
                for draw_index, indices in enumerate(model_indices)
            ]
        )
        report[profile]["summary"] = {
            "models": len(models),
            "positive_models": int(np.sum(point_gains > 0)),
            "mean_gain": round(float(point_gains.mean()), 4),
            "min_gain": round(float(point_gains.min()), 4),
            "max_gain": round(float(point_gains.max()), 4),
            "shield_minus_no_shield_mean": round(
                float(shield_delta_point_gains.mean()), 4
            ),
            "shield_minus_no_shield_min": round(
                float(shield_delta_point_gains.min()), 4
            ),
            "shield_positive_models": int(
                np.sum(shield_delta_point_gains > 0)
            ),
            "model_bootstrap_ci95": bootstrap_mean_ci(point_gains, rng),
            "fixed_models_image_ci95": [
                round(float(value), 4)
                for value in np.percentile(
                    model_draws.mean(axis=0), [2.5, 97.5]
                )
            ],
            "hierarchical_model_image_ci95": [
                round(float(value), 4)
                for value in np.percentile(hierarchical_draws, [2.5, 97.5])
            ],
            "no_shield": {
                "positive_models": int(np.sum(no_shield_point_gains > 0)),
                "mean_gain": round(float(no_shield_point_gains.mean()), 4),
                "min_gain": round(float(no_shield_point_gains.min()), 4),
                "max_gain": round(float(no_shield_point_gains.max()), 4),
                "fixed_models_image_ci95": [
                    round(float(value), 4)
                    for value in np.percentile(
                        no_shield_model_draws.mean(axis=0), [2.5, 97.5]
                    )
                ],
            },
            "shield_minus_no_shield": {
                "mean_delta": round(
                    float(shield_delta_point_gains.mean()), 4
                ),
                "min_delta": round(
                    float(shield_delta_point_gains.min()), 4
                ),
                "max_delta": round(
                    float(shield_delta_point_gains.max()), 4
                ),
                "fixed_models_image_ci95": [
                    round(float(value), 4)
                    for value in np.percentile(
                        shield_delta_model_draws.mean(axis=0), [2.5, 97.5]
                    )
                ],
            },
        }
        print(profile, "summary", report[profile]["summary"], flush=True)

    llvip_mean_draws = profile_draws["llvip"].mean(axis=0)
    m3fd_mean_draws = profile_draws["m3fd"].mean(axis=0)
    dataset_balanced_point = float(
        (
            np.mean(
                [
                    report["llvip"][model]["gain"]
                    for model in PROFILES["llvip"]
                ]
            )
            + np.mean(
                [
                    report["m3fd"][model]["gain"]
                    for model in PROFILES["m3fd"]
                ]
            )
        )
        / 2.0
    )
    dataset_balanced_draws = [
        (
            profile_draws["llvip"][
                rng.integers(0, len(PROFILES["llvip"]), len(PROFILES["llvip"])),
                draw_index,
            ].mean()
            + profile_draws["m3fd"][
                rng.integers(0, len(PROFILES["m3fd"]), len(PROFILES["m3fd"])),
                draw_index,
            ].mean()
        )
        / 2.0
        for draw_index in range(DRAWS)
    ]
    summary = {
        "policy": {"top_k": TOP_K, "visible_margin": MARGIN},
        "independent_datasets": list(PROFILES),
        "excluded_diagnostic": "dronevehicle official validation reuse",
        "test_models": len(all_gains),
        "positive_models": sum(gain > 0 for gain in all_gains),
        "mean_gain": round(float(np.mean(all_gains)), 4),
        "min_gain": round(float(np.min(all_gains)), 4),
        "max_gain": round(float(np.max(all_gains)), 4),
        "no_shield": {
            "positive_models": sum(gain > 0 for gain in all_no_shield_gains),
            "mean_gain": round(float(np.mean(all_no_shield_gains)), 4),
            "min_gain": round(float(np.min(all_no_shield_gains)), 4),
            "max_gain": round(float(np.max(all_no_shield_gains)), 4),
        },
        "shield_minus_no_shield": {
            "positive_models": sum(delta > 0 for delta in all_shield_deltas),
            "mean_delta": round(float(np.mean(all_shield_deltas)), 4),
            "min_delta": round(float(np.min(all_shield_deltas)), 4),
            "max_delta": round(float(np.max(all_shield_deltas)), 4),
        },
        "dataset_balanced": {
            "mean_gain": round(dataset_balanced_point, 4),
            "fixed_models_image_ci95": [
                round(float(value), 4)
                for value in np.percentile(
                    (llvip_mean_draws + m3fd_mean_draws) / 2.0,
                    [2.5, 97.5],
                )
            ],
            "hierarchical_model_image_ci95": [
                round(float(value), 4)
                for value in np.percentile(
                    dataset_balanced_draws, [2.5, 97.5]
                )
            ],
        },
        "profiles": report,
    }
    output = RESULTS / "confidence_shield_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
