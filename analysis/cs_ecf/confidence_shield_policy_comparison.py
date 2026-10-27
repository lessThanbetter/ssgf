"""Compare fixed fusion inputs with validation-frozen confidence shielding."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULTS = ROOT / "results/selective_fusion"
HELPERS = runpy.run_path(
    r"D:\Code\_server_handoff_v2\tmp\confidence_candidate_selection.py"
)
rows_for = HELPERS["rows_for"]
pooled = HELPERS["pooled"]
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
SUMMARY_PATH = RESULTS / "confidence_shield_summary.json"


def summarize_policy(rows, policy):
    gains = [rows[profile][model]["gain"] for profile, model in policy]
    return {
        "positive_models": int(sum(gain > 0 for gain in gains)),
        "mean_gain": round(float(np.mean(gains)), 4),
        "min_gain": round(float(np.min(gains)), 4),
        "max_gain": round(float(np.max(gains)), 4),
    }


def main():
    shield_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    report = {}
    flat = {}
    for profile, models in PROFILES.items():
        report[profile] = {}
        for model in models:
            rows = rows_for(profile, model)
            n_images = len(rows["vis"]["stems"])
            fixed = {
                candidate: pooled(rows, [candidate] * n_images)
                for candidate in rows
            }
            shield = shield_summary["profiles"][profile][model]
            report[profile][model] = {
                "ap50": {
                    "visible": round(fixed["vis"], 4),
                    "always_cdf": round(fixed["cdf"], 4),
                    "always_pia": round(fixed["pia"], 4),
                    "always_spg": round(fixed["spg"], 4),
                    "top5_no_shield": shield["visible_ap50"]
                    + shield["no_shield_gain"],
                    "confidence_shield": shield["selected_ap50"],
                },
                "gain": {
                    "visible": 0.0,
                    "always_cdf": round(fixed["cdf"] - fixed["vis"], 4),
                    "always_pia": round(fixed["pia"] - fixed["vis"], 4),
                    "always_spg": round(fixed["spg"] - fixed["vis"], 4),
                    "top5_no_shield": shield["no_shield_gain"],
                    "confidence_shield": shield["gain"],
                },
                "shield_selected_counts": shield["counts"],
            }
            flat[(profile, model)] = report[profile][model]["gain"]
            print(profile, model, report[profile][model], flush=True)

    policy_names = [
        "always_cdf",
        "always_pia",
        "always_spg",
        "top5_no_shield",
        "confidence_shield",
    ]
    summaries = {}
    for policy_name in policy_names:
        policy_rows = {
            profile: {
                model: {"gain": flat[(profile, model)][policy_name]}
                for model in models
            }
            for profile, models in PROFILES.items()
        }
        entries = [
            (profile, model)
            for profile, models in PROFILES.items()
            for model in models
        ]
        profile_means = [
            float(
                np.mean(
                    [
                        policy_rows[profile][model]["gain"]
                        for model in models
                    ]
                )
            )
            for profile, models in PROFILES.items()
        ]
        summaries[policy_name] = {
            **summarize_policy(policy_rows, entries),
            "dataset_balanced_mean_gain": round(
                float(np.mean(profile_means)), 4
            ),
            "profiles": {
                profile: summarize_policy(
                    policy_rows,
                    [(profile, model) for model in models],
                )
                for profile, models in PROFILES.items()
            },
        }

    output = {
        "policy": shield_summary["policy"],
        "independent_datasets": list(PROFILES),
        "excluded_diagnostic": shield_summary["excluded_diagnostic"],
        "models": report,
        "summaries": summaries,
    }
    output_path = RESULTS / "confidence_shield_policy_comparison.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print("WROTE", output_path)


if __name__ == "__main__":
    main()
