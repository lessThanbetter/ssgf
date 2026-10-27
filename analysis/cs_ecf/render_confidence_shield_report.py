"""Render the CS-ECF evidence tables directly from result JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULTS = ROOT / "results/selective_fusion"
OUTPUT = ROOT / "paper/CS_ECF_RESULTS.md"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def signed(value):
    return f"{value:+.4f}"


def interval(values):
    return f"`[{signed(values[0])},{signed(values[1])}]`"


def main():
    summary = load("confidence_shield_summary.json")
    three_dataset = load("confidence_shield_three_dataset_summary.json")
    group_summary = load("confidence_shield_group_hierarchical.json")
    thermal = load("ipt_thermal_stratification.json")
    mask_ablation = load("ipt_mask_ablation.json")
    dronevehicle = load("dronevehicle_test_combined.json")
    policies = load("confirmatory_policy_comparison.json")
    classes = load("confidence_shield_class_bootstrap.json")
    v2 = load("confidence_shield_v2_summary.json")
    v3 = load("confidence_shield_v3_summary.json")

    lines = [
        "# CS-ECF Results",
        "",
        "> Generated from `confidence_shield*.json`; do not edit tables by hand.",
        "",
        "## Frozen policy",
        "",
        f"- Top-k confidence: {summary['policy']['top_k']}",
        f"- Visible margin: {summary['policy']['visible_margin']}",
        "- Confirmatory datasets: LLVIP test, M3FD grouped test, "
        "DroneVehicle official test subset",
        "- DroneVehicle checkpoints: best and last from the same training run",
        "",
        "## Per-evaluator test results",
        "",
        "| Dataset | Evaluator | Visible AP50 | CS-ECF AP50 | Gain | Image CI95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for profile in ("llvip", "m3fd"):
        for model, row in summary["profiles"][profile].items():
            if model == "summary":
                continue
            lines.append(
                f"| {profile.upper()} | {model} | "
                f"{row['visible_ap50']:.4f} | {row['selected_ap50']:.4f} | "
                f"{signed(row['gain'])} | {interval(row['image_ci95'])} |"
            )
    for model, row in dronevehicle["checkpoint_results"].items():
        lines.append(
            f"| DRONEVEHICLE | {model} | "
            f"{row['visible_ap50']:.4f} | {row['shielded_ap50']:.4f} | "
            f"{signed(row['shielded_gain'])} | - |"
        )

    lines.extend(
        [
            "",
            "## Profile and dataset-balanced summaries",
            "",
            "| Scope | Models | Positive | Mean | Worst | Model CI95 | Image CI95 | Group/Hierarchical CI95 |",
            "|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for profile in ("llvip", "m3fd", "dronevehicle_test"):
        row = three_dataset[profile]["summary"]
        lines.append(
            f"| {profile.upper()} | {row['n_models']} | "
            f"{row['positive_models']}/{row['n_models']} | "
            f"{signed(row['mean_gain'])} | {signed(row['min_gain'])} | "
            f"{interval(row['model_ci95'])} | "
            f"{interval(row['image_ci95'])} | "
            f"{interval(group_summary[profile]['fixed_models_group_ci95'])} |"
        )
    aggregate = three_dataset["aggregate"]
    lines.append(
        f"| Dataset-balanced | {aggregate['n_checkpoints']} | "
        f"{aggregate['positive_checkpoints']}/{aggregate['n_checkpoints']} | "
        f"{signed(aggregate['dataset_balanced_mean'])} | "
        f"{signed(aggregate['min_checkpoint_gain'])} | "
        f"{interval(aggregate['dataset_balanced_model_ci95'])} | "
        f"{interval(aggregate['fixed_checkpoints_image_ci95'])} | "
        f"{interval(group_summary['aggregate']['dataset_balanced_hierarchical_group_ci95'])} |"
    )

    lines.extend(
        [
            "",
            "## DroneVehicle independent test",
            "",
            "| Checkpoint | Visible AP50 | CS-ECF AP50 | Shield gain | "
            "No-shield gain | Selected images |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, row in dronevehicle["checkpoint_results"].items():
        selected = sum(
            count
            for candidate, count in row["shielded_selected_counts"].items()
            if candidate != "vis"
        )
        lines.append(
            f"| {model} | {row['visible_ap50']:.4f} | "
            f"{row['shielded_ap50']:.4f} | "
            f"{signed(row['shielded_gain'])} | "
            f"{signed(row['no_shield_gain'])} | {selected}/300 |"
        )
    combined = dronevehicle["combined"]
    lines.append(
        f"| Combined | - | - | {signed(combined['mean_shielded_gain'])} | "
        f"{signed(combined['mean_no_shield_gain'])} | - |"
    )
    lines.extend(
        [
            "",
            f"- Image bootstrap 95% CI: {interval(combined['image_ci95'])}",
            f"- Group bootstrap 95% CI: {interval(combined['group_ci95'])}",
            "",
        ]
    )

    lines.extend(
        [
            "",
            "## Policy comparison",
            "",
            "| Policy | Positive checkpoints | Mean gain | Dataset-balanced | Worst |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "always_cdf": "Always CDF",
        "always_pia": "Always PIAFusion",
        "always_spg": "Always SPGFusion",
        "no_shield": "Top-5 no shield",
        "cs_ecf": "CS-ECF",
    }
    for policy in labels:
        row = policies["summaries"][policy]
        lines.append(
            f"| {labels[policy]} | {row['positive_checkpoints']}/12 | "
            f"{signed(row['mean_gain'])} | "
            f"{signed(row['dataset_balanced_mean'])} | "
            f"{signed(row['min_gain'])} |"
        )

    lines.extend(
        [
            "",
            "## Shield ablation",
            "",
            "| Scope | No-shield positive | No-shield mean | No-shield worst | "
            "Shield positive | Shield mean | Shield worst |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for profile in ("llvip", "m3fd"):
        row = summary["profiles"][profile]["summary"]
        lines.append(
            f"| {profile.upper()} | {row['no_shield']['positive_models']}/"
            f"{row['models']} | {signed(row['no_shield']['mean_gain'])} | "
            f"{signed(row['no_shield']['min_gain'])} | "
            f"{row['positive_models']}/{row['models']} | "
            f"{signed(row['mean_gain'])} | {signed(row['min_gain'])} |"
        )
    drone_no_shield = [
        row["no_shield_gain"]
        for row in dronevehicle["checkpoint_results"].values()
    ]
    drone_shield = [
        row["shielded_gain"]
        for row in dronevehicle["checkpoint_results"].values()
    ]
    lines.append(
        f"| DRONEVEHICLE_TEST | {sum(value > 0 for value in drone_no_shield)}/2 | "
        f"{signed(float(np.mean(drone_no_shield)))} | "
        f"{signed(min(drone_no_shield))} | "
        f"{sum(value > 0 for value in drone_shield)}/2 | "
        f"{signed(float(np.mean(drone_shield)))} | "
        f"{signed(min(drone_shield))} |"
    )
    no_shield = policies["summaries"]["no_shield"]
    shield = policies["summaries"]["cs_ecf"]
    lines.append(
        f"| Overall | {no_shield['positive_checkpoints']}/12 | "
        f"{signed(no_shield['mean_gain'])} | {signed(no_shield['min_gain'])} | "
        f"{shield['positive_checkpoints']}/12 | {signed(shield['mean_gain'])} | "
        f"{signed(shield['min_gain'])} |"
    )

    lines.extend(
        [
            "",
            "## M3FD class effects",
            "",
            "| Evaluator | Class | Visible AP50 | CS-ECF AP50 | Gain | Image CI95 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model, class_rows in classes["m3fd"].items():
        for class_id, row in class_rows.items():
            lines.append(
                f"| {model} | {row['class_name']} | "
                f"{row['visible_ap50']:.4f} | {row['shield_ap50']:.4f} | "
                f"{signed(row['gain'])} | {interval(row['image_ci95'])} |"
            )

    lines.extend(
        [
            "",
            "## Thermal association with switching",
            "",
            "| Dataset | Thermal rho (p) | Visible-luma rho (p) | "
            "CDF-mask rho (p) |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset, row in thermal.items():
        association = row["correlation_with_switch_fraction"]
        lines.append(
            f"| {dataset} | {association['thermal_rho']:+.4f} "
            f"({association['thermal_p']:.4f}) | "
            f"{association['visible_rho']:+.4f} "
            f"({association['visible_p']:.4f}) | "
            f"{association['thermal_mask_rho']:+.4f} "
            f"({association['thermal_mask_p']:.4f}) |"
        )

    lines.extend(
        [
            "",
            "## Thermal-mask ablation on DroneVehicle",
            "",
            "| Checkpoint | Visible | Masked CDF | Global CDF | "
            "Mask minus global |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("best", "last"):
        row = mask_ablation[model]
        lines.append(
            f"| {model} | {row['visible_ap50']:.4f} | "
            f"{row['thermal_mask_ap50']:.4f} | "
            f"{row['global_injection_ap50']:.4f} | "
            f"{signed(row['mask_minus_global'])} |"
        )

    lines.extend(
        [
            "",
            "## Rejected validation-selected variants",
            "",
            "| Variant | Candidates | Score | Margin | Test positive | Test mean | "
            "Dataset-balanced | Hierarchical CI95 |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for variant in (v2, v3):
        policy = variant["policy"]
        lines.append(
            f"| {policy['name']} | {', '.join(policy['candidates'])} | "
            f"{policy['score']} | {policy['visible_margin']:.2f} | "
            f"{variant['positive_models']}/{variant['test_models']} | "
            f"{signed(variant['mean_gain'])} | "
            f"{signed(variant['dataset_balanced']['mean_gain'])} | "
            f"{interval(variant['dataset_balanced']['hierarchical_model_image_ci95'])} |"
        )

    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- LLVIP intervals cross zero; do not describe the overall method as "
            "uniformly robust.",
            "- M3FD gain is concentrated in `people`; do not claim uniform "
            "improvement across six classes.",
            "- Always CDF remains a strong baseline (10/12 positive).",
            "- DroneVehicle image-level intervals cross zero; its three fixed "
            "groups are non-negative, but only two are materially positive.",
            "- Group-aware dataset-balanced CI is positive, but M3FD contributes "
            "only two groups and LLVIP seven scenes.",
            "- The two DroneVehicle checkpoints come from the same training run; "
            "they are not independent architectures.",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("WROTE", OUTPUT)


if __name__ == "__main__":
    main()
