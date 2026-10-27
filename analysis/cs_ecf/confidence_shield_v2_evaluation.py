"""Evaluate the validation-selected CS-ECF-v2 policy on held-out tests."""

from __future__ import annotations

import argparse
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
CANDIDATES = ("vis", "cdf", "pia")
TOP_K = 10
MARGIN = 1.3
SCORE_POLICY = "top"
POLICY_NAME = "CS-ECF-v2"
DRAWS = 400


def score(data, image_index):
    mask = data["img_idx"] == image_index
    confidence = np.sort(np.asarray(data["conf"][mask], np.float64))[::-1]
    if len(confidence) == 0:
        return -1.0
    if SCORE_POLICY == "mean":
        return float(confidence.mean())
    return float(confidence[:TOP_K].mean())


def choices_for(rows, n_images, margin):
    choices = []
    for image_index in range(n_images):
        values = {
            candidate: score(rows[candidate], image_index)
            for candidate in CANDIDATES
        }
        selected = max(values, key=values.get)
        if selected != "vis" and values[selected] < margin * values["vis"]:
            selected = "vis"
        choices.append(selected)
    return choices


def pooled_indices(rows, choices, indices):
    tp, conf, pcls, tcls = [], [], [], []
    n_gt = rows["vis"]["n_gt"]
    for image_index in indices:
        image_index = int(image_index)
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


def bootstrap_model_ci(values, rng):
    draws = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(DRAWS)
    ]
    return [
        round(float(value), 4)
        for value in np.percentile(draws, [2.5, 97.5])
    ]


def main():
    global CANDIDATES, TOP_K, MARGIN, SCORE_POLICY, POLICY_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument(
        "--score-policy", choices=("top", "mean"), default=SCORE_POLICY
    )
    parser.add_argument("--name", default=POLICY_NAME)
    parser.add_argument(
        "--output",
        default="confidence_shield_v2_summary.json",
    )
    args = parser.parse_args()
    CANDIDATES = tuple(
        candidate.strip()
        for candidate in args.candidates.split(",")
        if candidate.strip()
    )
    TOP_K = args.top_k
    MARGIN = args.margin
    SCORE_POLICY = args.score_policy
    POLICY_NAME = args.name
    rng = np.random.default_rng(4517)
    selection = json.loads(
        (RESULTS / "confidence_shield_validation_search.json").read_text(
            encoding="utf-8"
        )
    )
    fused_candidates = tuple(
        candidate for candidate in CANDIDATES if candidate != "vis"
    )
    candidate_name = (
        "cdf_only"
        if fused_candidates == ("cdf",)
        else "plus_".join(fused_candidates)
    )
    validation_score_name = (
        "mean" if SCORE_POLICY == "mean" else f"top{TOP_K}"
    )
    selection_record = next(
        record
        for record in selection["all_results"]
        if record["candidates"] == candidate_name
        and record["score_policy"] == validation_score_name
        and abs(record["visible_margin"] - MARGIN) < 1e-9
    )
    report = {}
    profile_draws = {}
    all_gains = []
    all_no_shield_gains = []
    for profile, models in PROFILES.items():
        report[profile] = {}
        model_draws = []
        no_shield_model_draws = []
        first_rows = rows_for(profile, models[0])
        n_images = len(first_rows["vis"]["stems"])
        bootstrap_indices = [
            rng.integers(0, n_images, n_images) for _ in range(DRAWS)
        ]
        for model in models:
            rows = rows_for(profile, model)
            choices = choices_for(rows, n_images, MARGIN)
            no_shield_choices = choices_for(rows, n_images, 1.0)
            baseline = pooled_indices(rows, ["vis"] * n_images, range(n_images))
            selected = pooled_indices(rows, choices, range(n_images))
            no_shield = pooled_indices(
                rows, no_shield_choices, range(n_images)
            )
            image_draws = []
            no_shield_image_draws = []
            for indices in bootstrap_indices:
                visible_draw = pooled_indices(
                    rows, ["vis"] * n_images, indices
                )
                image_draws.append(
                    pooled_indices(rows, choices, indices) - visible_draw
                )
                no_shield_image_draws.append(
                    pooled_indices(rows, no_shield_choices, indices)
                    - visible_draw
                )
            image_draws = np.asarray(image_draws)
            no_shield_image_draws = np.asarray(no_shield_image_draws)
            model_draws.append(image_draws)
            no_shield_model_draws.append(no_shield_image_draws)
            gain = selected - baseline
            no_shield_gain = no_shield - baseline
            all_gains.append(gain)
            all_no_shield_gains.append(no_shield_gain)
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
                "counts": {
                    candidate: choices.count(candidate)
                    for candidate in CANDIDATES
                },
            }
            print(profile, model, report[profile][model], flush=True)
        model_draws = np.asarray(model_draws)
        no_shield_model_draws = np.asarray(no_shield_model_draws)
        profile_draws[profile] = model_draws
        point_gains = np.asarray(
            [report[profile][model]["gain"] for model in models]
        )
        no_shield_point_gains = np.asarray(
            [
                report[profile][model]["no_shield_gain"]
                for model in models
            ]
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
            "model_bootstrap_ci95": bootstrap_model_ci(point_gains, rng),
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
                "positive_models": int(
                    np.sum(no_shield_point_gains > 0)
                ),
                "mean_gain": round(
                    float(no_shield_point_gains.mean()), 4
                ),
                "min_gain": round(
                    float(no_shield_point_gains.min()), 4
                ),
                "max_gain": round(
                    float(no_shield_point_gains.max()), 4
                ),
                "fixed_models_image_ci95": [
                    round(float(value), 4)
                    for value in np.percentile(
                        no_shield_model_draws.mean(axis=0), [2.5, 97.5]
                    )
                ],
            },
        }
        print(profile, "summary", report[profile]["summary"], flush=True)

    llvip_mean_draws = profile_draws["llvip"].mean(axis=0)
    m3fd_mean_draws = profile_draws["m3fd"].mean(axis=0)
    profile_points = [
        np.mean(
            [
                report[profile][model]["gain"]
                for model in PROFILES[profile]
            ]
        )
        for profile in PROFILES
    ]
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
        "policy": {
            "name": POLICY_NAME,
            "candidates": list(CANDIDATES),
            "score": (
                "mean_detection_confidence"
                if SCORE_POLICY == "mean"
                else f"top{TOP_K}_mean_detection_confidence"
            ),
            "visible_margin": MARGIN,
        },
        "validation_selection": selection_record,
        "test_models": len(all_gains),
        "positive_models": int(np.sum(np.asarray(all_gains) > 0)),
        "mean_gain": round(float(np.mean(all_gains)), 4),
        "min_gain": round(float(np.min(all_gains)), 4),
        "max_gain": round(float(np.max(all_gains)), 4),
        "no_shield": {
            "positive_models": int(
                np.sum(np.asarray(all_no_shield_gains) > 0)
            ),
            "mean_gain": round(float(np.mean(all_no_shield_gains)), 4),
            "min_gain": round(float(np.min(all_no_shield_gains)), 4),
            "max_gain": round(float(np.max(all_no_shield_gains)), 4),
        },
        "dataset_balanced": {
            "mean_gain": round(float(np.mean(profile_points)), 4),
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
        "decision": (
            "exploratory until confirmed on an independent dataset or "
            "untouched official test split"
        ),
    }
    output_path = RESULTS / args.output
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("WROTE", output_path)


if __name__ == "__main__":
    main()
