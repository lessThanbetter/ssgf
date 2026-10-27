"""Combine clean-split detector outputs into one auditable R2-2 summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import spearmanr


FUSED = [
    "mean",
    "activity_only",
    "tophat",
    "probe",
    "tophat_permuted",
    "probe_permuted",
    "tophat_wshuffle",
    "probe_wshuffle",
    "ensemble",
    "ifcnn",
    "densefuse",
    "csf",
    "piafusion",
    "spgfusion",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--multiplicity", required=True)
    parser.add_argument("--seed-ranges", required=True)
    parser.add_argument("--yolo11n-multiplicity")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scores = read_json(Path(args.scores))
    multiplicity = read_json(Path(args.multiplicity))
    seed_ranges = read_json(Path(args.seed_ranges))

    evaluators: dict[str, dict] = {}
    for family, label in (
        ("vis_split", "yolo11n_visible_clean"),
        ("agn_split", "yolo11n_modality_agnostic_clean"),
    ):
        source = seed_ranges[family]
        evaluators[label] = {
            "training_runs": len(source["runs"]),
            "run_tags": source["runs"],
            "visible_source_mAP50": source["visible_source_mAP50"],
            "inputs": {
                name: {
                    "mean_contrast": source["inputs"][name]["mean_contrast"],
                    "min_contrast": source["inputs"][name]["min_contrast"],
                    "max_contrast": source["inputs"][name]["max_contrast"],
                    "per_run_contrast": source["inputs"][name][
                        "per_run_contrast"
                    ],
                }
                for name in FUSED
            },
        }

    for detector, label in (
        ("yolo26n_clean", "yolo26n_visible_clean"),
        ("rtdetr_l_clean", "rtdetr_l_visible_clean"),
    ):
        if detector not in scores:
            raise SystemExit(f"missing detector in scores: {detector}")
        if detector not in multiplicity:
            raise SystemExit(f"missing detector in multiplicity: {detector}")
        tests = multiplicity[detector]["tests"]
        evaluators[label] = {
            "training_runs": 1,
            "run_tags": [detector],
            "visible_source_mAP50": scores[detector]["vis"]["mAP50"],
            "inputs": {
                name: {
                    "mAP50": scores[detector][name]["mAP50"],
                    "mean_contrast": scores[detector][name]["delta50_observed"],
                    "lo50": scores[detector][name]["lo50"],
                    "hi50": scores[detector][name]["hi50"],
                    "p_boot50": tests[name]["p_boot50"],
                    "q_bh50": tests[name]["q_bh50"],
                }
                for name in FUSED
            },
        }

    if args.yolo11n_multiplicity:
        multiplicity11 = read_json(Path(args.yolo11n_multiplicity))
        clean = multiplicity11["clean_visible"]["contours"]
        evaluators["yolo11n_visible_clean"]["reported_seed0"] = {
            name: {
                "mean_contrast": clean[name]["colour"]["delta50"],
                "lo50": clean[name]["colour"]["lo50"],
                "hi50": clean[name]["colour"]["hi50"],
                "p_boot50": clean[name]["colour"]["p"],
                "q_bh50": clean[name]["colour"]["q"],
            }
            for name in FUSED
        }

    summary = {}
    for label, evaluator in evaluators.items():
        inputs = evaluator["inputs"]
        summary[label] = {
            "below_direction": sum(row["mean_contrast"] < 0 for row in inputs.values()),
            "above_direction": sum(row["mean_contrast"] > 0 for row in inputs.values()),
            "min_contrast": min(row["mean_contrast"] for row in inputs.values()),
            "max_contrast": max(row["mean_contrast"] for row in inputs.values()),
        }
        if "q_bh50" in next(iter(inputs.values())):
            summary[label]["below_q05"] = sum(
                row["mean_contrast"] < 0 and row["q_bh50"] < 0.05
                for row in inputs.values()
            )
            summary[label]["above_q05"] = sum(
                row["mean_contrast"] > 0 and row["q_bh50"] < 0.05
                for row in inputs.values()
            )

    evaluator_names = list(evaluators)
    rank_associations = []
    for left_index, left in enumerate(evaluator_names):
        for right in evaluator_names[left_index + 1 :]:
            left_values = [
                evaluators[left]["inputs"][name]["mean_contrast"] for name in FUSED
            ]
            right_values = [
                evaluators[right]["inputs"][name]["mean_contrast"] for name in FUSED
            ]
            association = spearmanr(left_values, right_values)
            rank_associations.append(
                {
                    "left": left,
                    "right": right,
                    "n_inputs": len(FUSED),
                    "rho": float(association.statistic),
                    "p": float(association.pvalue),
                }
            )

    payload = {
        "protocol": {
            "dataset": "LLVIP",
            "split": {"train": 1350, "validation": 150, "test": 300},
            "test_labels": 744,
            "bootstrap_draws": 1000,
            "visible_source": "native colour image",
            "multiplicity": "Benjamini-Hochberg separately over 14 fused inputs for each single-run evaluator",
        },
        "summary": summary,
        "rank_associations": rank_associations,
        "evaluators": evaluators,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
