"""Merge the equal-group analysis into the canonical evidence files.

Historical pooled-image values are retained under explicit ``pooled_`` keys so
that the change of estimand is auditable rather than silently overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _map_interval(equal: dict, comparison: str) -> list[float]:
    return equal["dataset_balanced"][comparison][
        "run_cluster_by_group_ci95"
    ]


def _dataset_payload(equal_payload: dict, existing: dict) -> dict:
    comparisons = equal_payload["comparisons"]
    payload = {
        "point_csecf_minus_cdf": comparisons["csecf_minus_cdf"][
            "point_equal_group"
        ],
        "point_csecf_minus_visible": comparisons["csecf_minus_visible"][
            "point_equal_group"
        ],
        "point_cdf_only_minus_visible": comparisons[
            "cdf_only_minus_visible"
        ]["point_equal_group"],
        "point_csecf_minus_cdf_only": comparisons[
            "csecf_minus_cdf_only"
        ]["point_equal_group"],
        "fixed_models_group_ci95": comparisons["csecf_minus_cdf"][
            "run_cluster_by_group_ci95"
        ],
        "csecf_minus_visible_fixed_models_group_ci95": comparisons[
            "csecf_minus_visible"
        ]["run_cluster_by_group_ci95"],
        "cdf_only_minus_visible_fixed_models_group_ci95": comparisons[
            "cdf_only_minus_visible"
        ]["run_cluster_by_group_ci95"],
        "csecf_minus_cdf_only_fixed_models_group_ci95": comparisons[
            "csecf_minus_cdf_only"
        ]["run_cluster_by_group_ci95"],
    }
    payload.setdefault(
        "checkpoints",
        existing.get("checkpoints", []),
    )
    payload.setdefault(
        "run_clusters",
        existing.get("run_clusters", []),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equal", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--major", type=Path, required=True)
    parser.add_argument("--drone-combined", type=Path)
    args = parser.parse_args()

    equal = json.loads(args.equal.read_text(encoding="utf-8"))
    paired = json.loads(args.paired.read_text(encoding="utf-8"))
    major = json.loads(args.major.read_text(encoding="utf-8"))
    drone_combined = (
        json.loads(args.drone_combined.read_text(encoding="utf-8"))
        if args.drone_combined
        else None
    )

    source_note = {
        "source": args.equal.name,
        "point_and_interval_scope": equal["protocol"][
            "point_and_interval_scope"
        ],
    }
    paired["equal_group_source"] = source_note
    major["equal_group_source"] = source_note

    paired["bootstrap"]["interval"] = (
        "percentile 95% for the equal-group paired contrast"
    )
    paired["bootstrap"]["hierarchical_units"] = equal["protocol"][
        "hierarchical_summary"
    ]

    equal_by_path = equal["checkpoints"]
    for name, row in paired["per_checkpoint"].items():
        equal_key = f"{row['dataset']}/{name}"
        checkpoint = equal_by_path[equal_key]
        comparisons = checkpoint["comparisons"]
        updates = {
            "csecf_minus_cdf": ("csecf_minus_cdf", "point"),
            "csecf_minus_cdf_group_ci95": (
                "csecf_minus_cdf",
                "group_ci95",
            ),
            "cdf_only_gate_minus_visible": (
                "cdf_only_minus_visible",
                "point",
            ),
            "cdf_only_gate_minus_visible_group_ci95": (
                "cdf_only_minus_visible",
                "group_ci95",
            ),
            "csecf_minus_cdf_only_gate": (
                "csecf_minus_cdf_only",
                "point",
            ),
            "csecf_minus_cdf_only_gate_group_ci95": (
                "csecf_minus_cdf_only",
                "group_ci95",
            ),
            "csecf_minus_visible_group_point": (
                "csecf_minus_visible",
                "point",
            ),
            "csecf_minus_visible_group_ci95": (
                "csecf_minus_visible",
                "group_ci95",
            ),
            "cdf_minus_visible_group_point": (
                "cdf_minus_visible",
                "point",
            ),
            "cdf_minus_visible_group_ci95": (
                "cdf_minus_visible",
                "group_ci95",
            ),
        }
        for target, (comparison, field) in updates.items():
            value = comparisons[comparison][field]
            if target in row and f"pooled_{target}" not in row:
                row[f"pooled_{target}"] = row.get(target)
            row[target] = value

    for dataset, equal_payload in equal["per_dataset"].items():
        row = paired["per_dataset"][dataset]
        comparisons = equal_payload["comparisons"]
        fields = {
            "point_csecf_minus_cdf": "csecf_minus_cdf",
            "fixed_models_group_ci95": "csecf_minus_cdf",
            "point_csecf_minus_visible": "csecf_minus_visible",
            "csecf_minus_visible_fixed_models_group_ci95": (
                "csecf_minus_visible"
            ),
            "point_cdf_only_minus_visible": "cdf_only_minus_visible",
            "cdf_only_minus_visible_fixed_models_group_ci95": (
                "cdf_only_minus_visible"
            ),
            "point_csecf_minus_cdf_only": "csecf_minus_cdf_only",
            "csecf_minus_cdf_only_fixed_models_group_ci95": (
                "csecf_minus_cdf_only"
            ),
        }
        for target, comparison in fields.items():
            value = comparisons[comparison]
            replacement = (
                value["point_equal_group"]
                if target.startswith("point_")
                else value["run_cluster_by_group_ci95"]
            )
            if target in row and f"pooled_{target}" not in row:
                row[f"pooled_{target}"] = row.get(target)
            row[target] = replacement

    balanced = paired["dataset_balanced"]
    equal_balanced = equal["dataset_balanced"]
    balanced_fields = {
        "point_csecf_minus_cdf": "csecf_minus_cdf",
        "run_cluster_by_group_ci95": "csecf_minus_cdf",
        "point_csecf_minus_visible": "csecf_minus_visible",
        "visible_run_cluster_by_group_ci95": "csecf_minus_visible",
        "point_cdf_only_minus_visible": "cdf_only_minus_visible",
        "cdf_only_run_cluster_by_group_ci95": "cdf_only_minus_visible",
        "point_csecf_minus_cdf_only": "csecf_minus_cdf_only",
        "csecf_minus_cdf_only_run_cluster_by_group_ci95": (
            "csecf_minus_cdf_only"
        ),
    }
    for target, comparison in balanced_fields.items():
        payload = equal_balanced[comparison]
        replacement = (
            payload["point_equal_group"]
            if target.startswith("point_")
            else payload["run_cluster_by_group_ci95"]
        )
        if target in balanced and f"pooled_{target}" not in balanced:
            balanced[f"pooled_{target}"] = balanced.get(target)
        balanced[target] = replacement
    balanced["stability_ci95_equal_group_endpoint_range"] = {
        key: payload["stability_ci95_endpoint_range"]
        for key, payload in equal_balanced.items()
    }

    for name, row in major["checkpoints"].items():
        checkpoint = equal_by_path[name]["comparisons"]
        for target, comparison in (
            ("visible_delta", "csecf_minus_visible"),
            ("cdf_delta", "csecf_minus_cdf"),
        ):
            if target in row and f"pooled_{target}" not in row:
                row[f"pooled_{target}"] = row.get(target)
            row[target] = checkpoint[comparison]["point"]
        for target, comparison in (
            ("visible_group_ci95", "csecf_minus_visible"),
            ("cdf_group_ci95", "csecf_minus_cdf"),
        ):
            if target in row and f"pooled_{target}" not in row:
                row[f"pooled_{target}"] = row.get(target)
            row[target] = checkpoint[comparison]["group_ci95"]

    major["protocol"]["independent_unit"] = (
        "paired evaluation group for the equal-group estimand; detector "
        "checkpoint within training-run cluster for hierarchical summaries"
    )
    major["protocol"]["primary_summary"] = equal["protocol"][
        "primary_estimand"
    ]

    visible = equal_balanced["csecf_minus_visible"]
    cdf = equal_balanced["csecf_minus_cdf"]
    major["dataset_balanced"]["visible"] = {
        "point": visible["point_equal_group"],
        "hierarchical_ci95": visible["run_cluster_by_group_ci95"],
        "stability_ci95_endpoint_range": visible[
            "stability_ci95_endpoint_range"
        ],
        "estimand": "equal-group paired AP50 difference",
    }
    major["dataset_balanced"]["cdf"] = {
        "point": cdf["point_equal_group"],
        "hierarchical_ci95": cdf["run_cluster_by_group_ci95"],
        "stability_ci95_endpoint_range": cdf[
            "stability_ci95_endpoint_range"
        ],
        "estimand": "equal-group paired AP50 difference",
    }
    major["cdf_comparison"]["dataset_balanced"] = {
        "point_csecf_minus_cdf": cdf["point_equal_group"],
        "run_cluster_by_group_ci95": cdf[
            "run_cluster_by_group_ci95"
        ],
    }
    if "per_dataset" in major["cdf_comparison"]:
        major["cdf_comparison"].setdefault(
            "pooled_per_dataset",
            major["cdf_comparison"]["per_dataset"],
        )
    major["cdf_comparison"]["per_dataset"] = {
        dataset: _dataset_payload(
            payload,
            major["cdf_comparison"].get("pooled_per_dataset", {}).get(
                dataset,
                {},
            ),
        )
        for dataset, payload in equal["per_dataset"].items()
    }
    major["cdf_only_gate"]["dataset_balanced"] = {
        "point_cdf_only_minus_visible": equal_balanced[
            "cdf_only_minus_visible"
        ]["point_equal_group"],
        "interval": _map_interval(equal, "cdf_only_minus_visible"),
        "point_csecf_minus_cdf_only": equal_balanced[
            "csecf_minus_cdf_only"
        ]["point_equal_group"],
        "csecf_minus_cdf_only_interval": _map_interval(
            equal,
            "csecf_minus_cdf_only",
        ),
    }
    if "per_dataset" in major["cdf_only_gate"]:
        major["cdf_only_gate"].setdefault(
            "pooled_per_dataset",
            major["cdf_only_gate"]["per_dataset"],
        )
    major["cdf_only_gate"]["per_dataset"] = {
        dataset: {
            "point_cdf_only_minus_visible": payload["comparisons"][
                "cdf_only_minus_visible"
            ]["point_equal_group"],
            "interval": payload["comparisons"][
                "cdf_only_minus_visible"
            ]["run_cluster_by_group_ci95"],
            "point_csecf_minus_cdf_only": payload["comparisons"][
                "csecf_minus_cdf_only"
            ]["point_equal_group"],
            "csecf_minus_cdf_only_interval": payload["comparisons"][
                "csecf_minus_cdf_only"
            ]["run_cluster_by_group_ci95"],
        }
        for dataset, payload in equal["per_dataset"].items()
    }
    major["equal_group_policy_summary"] = equal["policy_summary"]
    major["ap50_95_sensitivity"] = equal["ap50_95_sensitivity"]
    major["test_policy_grid_equal_group"] = equal[
        "test_policy_grid_equal_group"
    ]
    csecf_policy = equal["policy_summary"]["csecf"]
    central_grid = major["test_policy_grid"]["K5_m1.5"]
    central_grid["dataset_balanced_mean"] = csecf_policy[
        "dataset_balanced_equal_group_gain"
    ]
    central_grid["mean_checkpoint_gain"] = float(
        sum(
            csecf_policy["checkpoint_equal_group_gains"].values()
        )
        / len(csecf_policy["checkpoint_equal_group_gains"])
    )
    central_grid["min_checkpoint_gain"] = csecf_policy[
        "worst_checkpoint_gain"
    ]
    central_grid["positive_checkpoints"] = csecf_policy[
        "positive_checkpoints"
    ]
    central_grid["estimand"] = (
        "equal-group point estimates; tested only for the frozen policy"
    )
    major["baselines"]["estimand_note"] = (
        "Control rows retain pooled-checkpoint calculations. They are "
        "descriptive and are not used as matched evidence for the "
        "equal-group primary contrast."
    )

    if drone_combined is not None:
        drone_combined.setdefault("pooled_combined", drone_combined.get("combined", {}))
        drone_combined["independent_status"] = (
            "Official test subset scored with native OBB AP; best and last "
            "checkpoints share one upstream training run."
        )
        drone_combined["metric"] = (
            "Native oriented-box AP50 using probabilistic rotated-box IoU "
            "and one-to-one maximum-cardinality matching."
        )
        drone_combined["estimand_note"] = (
            "checkpoint_results retain pooled-image AP values. The combined "
            "object is the canonical equal-group paired contrast used in the "
            "manuscript."
        )
        drone_combined["combined"] = {
            "estimand": "equal-group paired AP50 difference",
            "equal_group_csecf_minus_visible": {
                "point": equal["per_dataset"]["dronevehicle_test"][
                    "comparisons"
                ]["csecf_minus_visible"]["point_equal_group"],
                "group_ci95": equal["per_dataset"]["dronevehicle_test"][
                    "comparisons"
                ]["csecf_minus_visible"][
                    "run_cluster_by_group_ci95"
                ],
            },
            "equal_group_csecf_minus_cdf": {
                "point": equal["per_dataset"]["dronevehicle_test"][
                    "comparisons"
                ]["csecf_minus_cdf"]["point_equal_group"],
                "group_ci95": equal["per_dataset"]["dronevehicle_test"][
                    "comparisons"
                ]["csecf_minus_cdf"]["run_cluster_by_group_ci95"],
            },
            "equal_group_cdf_minus_visible": {
                "point": equal["per_dataset"]["dronevehicle_test"][
                    "comparisons"
                ]["cdf_minus_visible"]["point_equal_group"],
                "group_ci95": equal["per_dataset"]["dronevehicle_test"][
                    "comparisons"
                ]["cdf_minus_visible"]["run_cluster_by_group_ci95"],
            },
            "equal_group_no_shield_minus_visible": {
                "point": equal["per_dataset"]["dronevehicle_test"][
                    "comparisons"
                ]["no_shield_minus_visible"]["point_equal_group"],
                "group_ci95": equal["per_dataset"]["dronevehicle_test"][
                    "comparisons"
                ]["no_shield_minus_visible"][
                    "run_cluster_by_group_ci95"
                ],
            },
        }

    _write(args.paired, paired)
    _write(args.major, major)
    if drone_combined is not None and args.drone_combined is not None:
        _write(args.drone_combined, drone_combined)
        print("UPDATED", args.drone_combined)
    print("UPDATED", args.paired)
    print("UPDATED", args.major)


if __name__ == "__main__":
    main()
