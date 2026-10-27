"""Search confidence-shield variants using validation data only."""

from __future__ import annotations

import itertools
import json
import runpy
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULTS = ROOT / "results/selective_fusion"
HELPERS = runpy.run_path(
    r"D:\Code\_server_handoff_v2\tmp\confidence_margin_validation.py"
)
load_rows = HELPERS["load_rows"]
pooled = HELPERS["pooled"]
PROFILES = {
    "llvip": ["yolo26n", "rtdetr"],
    "m3fd": ["yolo26_m3fd6", "yolo8n_m3fd6"],
}
FUSED_CANDIDATES = ("cdf", "pia", "spg")
SCORE_POLICIES = ("top1", "top3", "top5", "top10", "mean", "sum_log")
MARGINS = (1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0)


def confidence_score(data, image_index, policy):
    mask = data["img_idx"] == image_index
    confidence = np.sort(np.asarray(data["conf"][mask], np.float64))[::-1]
    if len(confidence) == 0:
        return -1.0
    if policy == "top1":
        return float(confidence[0])
    if policy.startswith("top"):
        top_k = int(policy[3:])
        return float(confidence[:top_k].mean())
    if policy == "mean":
        return float(confidence.mean())
    if policy == "sum_log":
        return float(np.log1p(confidence).sum())
    raise ValueError(policy)


def subset_name(candidates):
    return "cdf_only" if candidates == ("cdf",) else "plus_".join(candidates)


def main():
    rows = {
        (profile, model): load_rows(profile, model)
        for profile, models in PROFILES.items()
        for model in models
    }
    baseline = {}
    for key, model_rows in rows.items():
        n_images = len(model_rows["vis"]["stems"])
        baseline[key] = pooled(model_rows, ["vis"] * n_images)

    results = []
    subsets = [
        subset
        for size in range(1, len(FUSED_CANDIDATES) + 1)
        for subset in itertools.combinations(FUSED_CANDIDATES, size)
    ]
    for subset in subsets:
        for score_policy in SCORE_POLICIES:
            for margin in MARGINS:
                model_gains = {}
                for key, model_rows in rows.items():
                    profile, model = key
                    n_images = len(model_rows["vis"]["stems"])
                    choices = []
                    for image_index in range(n_images):
                        values = {
                            candidate: confidence_score(
                                model_rows[candidate],
                                image_index,
                                score_policy,
                            )
                            for candidate in ("vis", *subset)
                        }
                        selected = max(values, key=values.get)
                        if (
                            selected != "vis"
                            and values[selected]
                            < margin * values["vis"]
                        ):
                            selected = "vis"
                        choices.append(selected)
                    selected_ap = pooled(model_rows, choices)
                    model_gains[key] = selected_ap - baseline[key]
                profile_means = {
                    profile: float(
                        np.mean([model_gains[(profile, model)] for model in models])
                    )
                    for profile, models in PROFILES.items()
                }
                gains = np.asarray(list(model_gains.values()))
                results.append(
                    {
                        "candidates": subset_name(subset),
                        "score_policy": score_policy,
                        "visible_margin": margin,
                        "model_gains": {
                            f"{profile}/{model}": round(gain, 6)
                            for (profile, model), gain in model_gains.items()
                        },
                        "profile_means": {
                            profile: round(gain, 6)
                            for profile, gain in profile_means.items()
                        },
                        "dataset_balanced_mean": round(
                            float(np.mean(list(profile_means.values()))), 6
                        ),
                        "model_mean": round(float(gains.mean()), 6),
                        "min_gain": round(float(gains.min()), 6),
                        "max_gain": round(float(gains.max()), 6),
                        "positive_models": int(np.sum(gains > 0)),
                    }
                )
                print(
                    subset_name(subset),
                    score_policy,
                    margin,
                    "balanced",
                    results[-1]["dataset_balanced_mean"],
                    "min",
                    results[-1]["min_gain"],
                    flush=True,
                )

    feasible = [
        result for result in results if result["min_gain"] >= -0.0005
    ]
    best_balanced_feasible = max(
        feasible, key=lambda result: result["dataset_balanced_mean"]
    )
    best_worst_case = max(results, key=lambda result: result["min_gain"])
    best_balanced = max(
        results, key=lambda result: result["dataset_balanced_mean"]
    )
    output = {
        "selection_data": "validation only",
        "objective": (
            "maximize dataset-balanced validation gain subject to "
            "every validation evaluator losing no more than 0.0005 AP50"
        ),
        "best_balanced_feasible": best_balanced_feasible,
        "best_worst_case": best_worst_case,
        "best_balanced_unconstrained": best_balanced,
        "top_feasible": sorted(
            feasible,
            key=lambda result: result["dataset_balanced_mean"],
            reverse=True,
        )[:20],
        "all_results": results,
    }
    output_path = RESULTS / "confidence_shield_validation_search.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: output[key]
                for key in (
                    "best_balanced_feasible",
                    "best_worst_case",
                    "best_balanced_unconstrained",
                )
            },
            indent=2,
        )
    )
    print("WROTE", output_path)


if __name__ == "__main__":
    main()
