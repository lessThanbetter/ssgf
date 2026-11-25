"""Summarize CDF parameter sensitivity on the M3FD validation split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    path
    for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
    if (path / "scripts" / "local_ap_reference.py").is_file()
)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis" / "cs_ecf"))

import budgeted_candidate_selection as budget  # noqa: E402
from local_ap_reference import load  # noqa: E402


MODELS = ("yolo26_m3fd6", "yolo8n_m3fd6")
BASELINE = "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"
PACKAGE_PARENT = ROOT / "paper" / "CS_ECF_IPT_Submission"
DEFAULT_PACKAGE = next(
    (
        path
        for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
        if (path / "main.tex").is_file()
        and (path / "evidence").is_dir()
    ),
    max(
        PACKAGE_PARENT.glob("submission_package_v*"),
        key=lambda path: int(path.name.rsplit("v", 1)[1]),
    ),
)
PACKAGED = (
    (DEFAULT_PACKAGE / "evidence" / "m3fd_cdf_validation").is_dir()
    and (DEFAULT_PACKAGE / "evidence" / "m3fd_cdf_test").is_dir()
)
CDF_VARIANTS = {
    "baseline": ("base", "q=0.95, alpha=0.20, sigma=5, min=0"),
    "q090": ("q", "q=0.90, alpha=0.20, sigma=5"),
    "q099": ("q", "q=0.99, alpha=0.20, sigma=5"),
    "alpha010": ("alpha", "q=0.95, alpha=0.10, sigma=5"),
    "alpha030": ("alpha", "q=0.95, alpha=0.30, sigma=5"),
    "sigma03": ("sigma", "q=0.95, alpha=0.20, sigma=3"),
    "sigma07": ("sigma", "q=0.95, alpha=0.20, sigma=7"),
    "min10": ("magnitude", "q=0.95, alpha=0.20, sigma=5, min=1"),
    "min20": ("magnitude", "q=0.95, alpha=0.20, sigma=5, min=2"),
    "min40": ("magnitude", "q=0.95, alpha=0.20, sigma=5, min=4"),
}


def cdf_path(
    variant: str,
    sensitivity_root: Path,
    baseline_root: Path,
) -> Path:
    if variant == "baseline":
        return baseline_root / f"{MODELS[0]}__{BASELINE}.npz"
    if variant.startswith("q"):
        threshold = {"q090": "q090", "q099": "q099"}[variant]
        return (
            sensitivity_root
            / f"{MODELS[0]}__consensus_intersection_pixel_mean_ifcnn_"
            f"{threshold}_a020_s05.npz"
        )
    if variant.startswith("alpha"):
        alpha = {"alpha010": "a010", "alpha030": "a030"}[variant]
        return (
            sensitivity_root
            / f"{MODELS[0]}__consensus_intersection_pixel_mean_ifcnn_"
            f"q095_{alpha}_s05.npz"
        )
    if variant.startswith("sigma"):
        sigma = {"sigma03": "s03", "sigma07": "s07"}[variant]
        return (
            sensitivity_root
            / f"{MODELS[0]}__consensus_intersection_pixel_mean_ifcnn_"
            f"q095_a020_{sigma}.npz"
        )
    suffix = {"min10": "m10", "min20": "m20", "min40": "m40"}[variant]
    return baseline_root / f"{MODELS[0]}__{BASELINE}_{suffix}.npz"


def states_for_model(
    model: str,
    variant: str,
    baseline_root: Path,
    sensitivity_root: Path,
    sota_root: Path,
) -> dict[str, dict[str, Any]]:
    variant_path = cdf_path(variant, sensitivity_root, baseline_root)
    variant_path = variant_path.with_name(
        variant_path.name.replace(MODELS[0], model, 1)
    )
    if not variant_path.is_file():
        raise FileNotFoundError(variant_path)
    visible_path = baseline_root / f"{model}__vis.npz"
    pia_path = sota_root / f"{model}__piafusion.npz"
    spg_path = sota_root / f"{model}__spgfusion.npz"
    return {
        "vis": load(visible_path),
        "cdf": load(variant_path),
        "pia": load(pia_path),
        "spg": load(spg_path),
    }


def evaluate_variant(
    variant: str,
    baseline_root: Path,
    sensitivity_root: Path,
    sota_root: Path,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    full_gains = []
    selected_gains = []
    cdf_deltas = []
    for model in MODELS:
        states = states_for_model(
            model,
            variant,
            baseline_root,
            sensitivity_root,
            sota_root,
        )
        indices = np.arange(int(states["vis"]["n"]))
        visible_ap = budget.pooled_ap50(states["vis"], indices)
        cdf_ap = budget.pooled_ap50(states["cdf"], indices)
        full_policy = budget.Policy(budget.CANDIDATES, budget.FROZEN_MARGIN)
        selected_policy = budget.Policy(("vis", "pia"), budget.FROZEN_MARGIN)
        full_stats = budget.selected_stats(
            states,
            budget.candidate_choices(states, full_policy),
        )
        selected_stats = budget.selected_stats(
            states,
            budget.candidate_choices(states, selected_policy),
        )
        full_gain = (
            budget.pooled_ap50(full_stats, indices) - visible_ap
        )
        selected_gain = (
            budget.pooled_ap50(selected_stats, indices) - visible_ap
        )
        cdf_delta = cdf_ap - visible_ap
        full_gains.append(full_gain)
        selected_gains.append(selected_gain)
        cdf_deltas.append(cdf_delta)
        rows[model] = {
            "visible_ap50": visible_ap,
            "cdf_ap50": cdf_ap,
            "cdf_minus_visible": cdf_delta,
            "full_policy_minus_visible": full_gain,
            "selected_policy_minus_visible": selected_gain,
        }
    return {
        "variant": variant,
        "parameter": CDF_VARIANTS[variant][1],
        "mean_cdf_minus_visible": float(np.mean(cdf_deltas)),
        "mean_full_policy_minus_visible": float(np.mean(full_gains)),
        "worst_full_policy_minus_visible": float(np.min(full_gains)),
        "mean_selected_policy_minus_visible": float(np.mean(selected_gains)),
        "positive_full_policy_evaluators": int(
            sum(value > 0 for value in full_gains)
        ),
        "models": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=(
            DEFAULT_PACKAGE / "evidence" / "m3fd_cdf_validation"
            if PACKAGED
            else ROOT / "results" / "selective_fusion"
            / "m3fd_tau_validation_6cls"
        ),
    )
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=(
            DEFAULT_PACKAGE / "evidence" / "cdf_sensitivity"
            / "validation_scores"
            if PACKAGED
            else ROOT / "results" / "selective_fusion"
            / "cdf_sensitivity_scores"
        ),
    )
    parser.add_argument(
        "--sota-root",
        type=Path,
        default=(
            DEFAULT_PACKAGE / "evidence" / "m3fd_sota_validation"
            if PACKAGED
            else ROOT / "results" / "selective_fusion"
            / "m3fd_sota_validation"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            DEFAULT_PACKAGE / "evidence" / "cdf_sensitivity"
            if SCRIPT_PATH.parent.name == "reproduction"
            else ROOT / "analysis" / "cs_ecf" / "results"
            / "cdf_sensitivity"
        ),
    )
    parser.add_argument(
        "--test-baseline-root",
        type=Path,
        default=(
            DEFAULT_PACKAGE / "evidence" / "m3fd_cdf_test"
            if PACKAGED
            else ROOT / "results" / "selective_fusion"
            / "m3fd_tau_test_6cls"
        ),
    )
    parser.add_argument(
        "--test-sensitivity-root",
        type=Path,
        default=(
            DEFAULT_PACKAGE / "evidence" / "cdf_sensitivity"
            / "test_scores"
            if PACKAGED
            else ROOT / "results" / "selective_fusion"
            / "cdf_sensitivity_test_scores"
        ),
    )
    parser.add_argument(
        "--test-sota-root",
        type=Path,
        default=(
            DEFAULT_PACKAGE / "evidence" / "m3fd_sota_test"
            if PACKAGED
            else ROOT / "results" / "selective_fusion"
            / "m3fd_sota_scores"
        ),
    )
    args = parser.parse_args()
    validation_rows = [
        evaluate_variant(
            variant,
            args.baseline_root,
            args.sensitivity_root,
            args.sota_root,
        )
        for variant in CDF_VARIANTS
    ]
    test_rows = [
        evaluate_variant(
            variant,
            args.test_baseline_root,
            args.test_sensitivity_root,
            args.test_sota_root,
        )
        for variant in CDF_VARIANTS
    ]
    payload = {
        "matching": "all-at-once maximum-cardinality matching",
        "top_k": budget.TOP_K,
        "margin": budget.FROZEN_MARGIN,
        "baseline": BASELINE,
        "parameters": {
            variant: parameter for variant, (_, parameter) in CDF_VARIANTS.items()
        },
        "validation": {
            "scope": (
                "M3FD validation, two six-class visible-trained evaluators"
            ),
            "rows": validation_rows,
        },
        "test": {
            "scope": "M3FD held-out test, two six-class visible-trained evaluators",
            "rows": test_rows,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.out / "sensitivity.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        fields = (
            "scope",
            "variant",
            "parameter",
            "mean_cdf_minus_visible",
            "mean_full_policy_minus_visible",
            "worst_full_policy_minus_visible",
            "mean_selected_policy_minus_visible",
            "positive_full_policy_evaluators",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scope, rows in (
            ("validation", validation_rows),
            ("test", test_rows),
        ):
            for row in rows:
                writer.writerow({
                    "scope": scope,
                    **{field: row[field] for field in fields if field != "scope"},
                })
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
