"""Recompute paired policy contrasts with an explicit equal-group estimand.

The original paired analysis attached a pooled-image point estimate to a
group-resampling interval. This script removes that mismatch. Each group is
treated as one observation, checkpoints are averaged within a training-run
cluster, and LLVIP, M3FD, and DroneVehicle are then weighted equally.

The script also records pooled AP50 and AP50-95 descriptive values, but those
are not used as the target of the group-resampling intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("SSGF_ROOT", str(HERE.parents[1])))
RESULTS_ROOT = Path(
    os.environ.get(
        "SSGF_RESULTS_ROOT",
        str(DATA_ROOT / "results" / "selective_fusion"),
    )
)
DEPOSIT_ROOT = Path(
    os.environ.get(
        "SSGF_DEPOSIT_ROOT",
        str(
            DATA_ROOT
            / "_deposit"
            / "ssgf-llvip-fusion-detection-v1"
            / "data"
            / "per_image"
        ),
    )
)

os.environ.setdefault("SSGF_ROOT", str(DATA_ROOT))
os.environ.setdefault("SSGF_RESULTS_ROOT", str(RESULTS_ROOT))
os.environ.setdefault("SSGF_DEPOSIT_ROOT", str(DEPOSIT_ROOT))

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DATA_ROOT / "scripts"))

import analyze_csecf_vs_cdf as csecf  # noqa: E402
from local_ap_reference import pooled  # noqa: E402


DATASETS = ("llvip", "m3fd", "dronevehicle_test")
COMPARISONS = {
    "csecf_minus_visible": ("csecf", "visible"),
    "csecf_minus_cdf": ("csecf", "cdf"),
    "cdf_minus_visible": ("cdf", "visible"),
    "cdf_only_minus_visible": ("cdf_only", "visible"),
    "csecf_minus_cdf_only": ("csecf", "cdf_only"),
    "no_shield_minus_visible": ("no_shield", "visible"),
}
CHECKPOINT_DRAWS = 2000
AGGREGATE_DRAWS = 5000
STABILITY_RUNS = 20
STABILITY_DRAWS = 5000
SEED = 2219


def _ap(stats: dict, group_indices: np.ndarray) -> float:
    return float(pooled(stats, group_indices)[0])


def _point_metrics(stats: dict) -> dict[str, float]:
    ap50, ap5095 = pooled(stats, np.arange(stats["n"]))
    return {"ap50": float(ap50), "ap50_95": float(ap5095)}


def _checkpoint_record(dataset: str, name: str, run_cluster: str,
                       states: dict, rng: np.random.Generator) -> dict:
    shielded, unshielded = csecf._decisions(states)
    cdf_only, _ = csecf._decisions(
        states,
        candidates=("vis", "cdf"),
        top_k=csecf.TOP_K,
        margin=csecf.VISIBLE_MARGIN,
    )
    policies = {
        "visible": states["vis"],
        "cdf": states["cdf"],
        "pia": states["pia"],
        "spg": states["spg"],
        "no_shield": csecf._policy_stats(states, unshielded),
        "csecf": csecf._policy_stats(states, shielded),
        "cdf_only": csecf._policy_stats(states, cdf_only),
    }
    groups = csecf._groups(dataset, [str(s) for s in states["vis"]["stems"]])
    group_metrics = {
        policy: np.asarray(
            [_ap(stats, group) for group in groups],
            dtype=float,
        )
        for policy, stats in policies.items()
    }
    policy_group_gains = {
        policy: values - group_metrics["visible"]
        for policy, values in group_metrics.items()
    }
    comparisons = {
        key: group_metrics[left] - group_metrics[right]
        for key, (left, right) in COMPARISONS.items()
    }
    checkpoint_intervals = {}
    for key, values in comparisons.items():
        draws = np.asarray(
            [
                values[
                    rng.integers(0, len(values), size=len(values))
                ].mean()
                for _ in range(CHECKPOINT_DRAWS)
            ]
        )
        checkpoint_intervals[key] = {
            "point": float(values.mean()),
            "group_ci95": [
                float(value)
                for value in np.percentile(draws, [2.5, 97.5])
            ],
        }
    policy_grid = {}
    for top_k in (1, 3, 5, 10):
        for margin in (1.2, 1.5, 2.0):
            choices, _ = csecf._decisions(
                states,
                top_k=top_k,
                margin=margin,
            )
            stats = csecf._policy_stats(states, choices)
            gains = np.asarray([
                _ap(stats, group) - _ap(states["vis"], group)
                for group in groups
            ])
            policy_grid[f"K{top_k}_m{margin:.1f}"] = {
                "gain": float(gains.mean()),
                "switch_fraction": (
                    sum(choice != "vis" for choice in choices)
                    / len(choices)
                ),
            }
    return {
        "dataset": dataset,
        "name": name,
        "run_cluster": run_cluster,
        "n_images": states["vis"]["n"],
        "n_groups": len(groups),
        "policy_metrics": {
            policy: _point_metrics(stats)
            for policy, stats in policies.items()
        },
        "selected_counts": {
            candidate: shielded.count(candidate)
            for candidate in ("vis", "cdf", "pia", "spg")
        },
        "comparisons": checkpoint_intervals,
        "policy_group_gains": {
            policy: [float(value) for value in values]
            for policy, values in policy_group_gains.items()
        },
        "policy_grid_equal_group": policy_grid,
        "_group_values": comparisons,
        "_policy_group_gains": policy_group_gains,
    }


def _cluster_means(rows: list[dict], key: str) -> np.ndarray:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[row["run_cluster"]].append(
            float(row["_group_values"][key].mean())
        )
    return np.asarray(
        [float(np.mean(values)) for values in by_cluster.values()],
        dtype=float,
    )


def _dataset_point(rows: list[dict], key: str) -> float:
    return float(_cluster_means(rows, key).mean())


def _dataset_policy_point(rows: list[dict], policy: str) -> float:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[row["run_cluster"]].append(
            float(row["_policy_group_gains"][policy].mean())
        )
    return float(
        np.mean([
            float(np.mean(values))
            for values in by_cluster.values()
        ])
    )


def _dataset_draws(rows: list[dict], draws: int,
                   seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    group_count = len(next(iter(rows))["_group_values"][
        "csecf_minus_visible"
    ])
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cluster[row["run_cluster"]].append(row)
    cluster_names = sorted(by_cluster)
    output = {key: [] for key in COMPARISONS}
    for _ in range(draws):
        sampled_groups = rng.integers(
            0, group_count, size=group_count
        )
        sampled_clusters = rng.integers(
            0,
            len(cluster_names),
            size=len(cluster_names),
        )
        for key in COMPARISONS:
            cluster_values = []
            for cluster_index in sampled_clusters:
                model_values = [
                    float(
                        row["_group_values"][key][
                            sampled_groups
                        ].mean()
                    )
                    for row in by_cluster[
                        cluster_names[int(cluster_index)]
                    ]
                ]
                cluster_values.append(float(np.mean(model_values)))
            output[key].append(float(np.mean(cluster_values)))
    return {
        key: np.asarray(values, dtype=float)
        for key, values in output.items()
    }


def _hierarchical_draws(by_dataset: dict[str, list[dict]],
                        draws: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    outputs = {key: [] for key in COMPARISONS}
    for _ in range(draws):
        dataset_draws = {key: [] for key in COMPARISONS}
        for dataset in DATASETS:
            rows = by_dataset[dataset]
            group_count = len(next(iter(rows))["_group_values"][
                "csecf_minus_visible"
            ])
            sampled_groups = rng.integers(
                0, group_count, size=group_count
            )
            by_cluster: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                by_cluster[row["run_cluster"]].append(row)
            cluster_names = sorted(by_cluster)
            sampled_clusters = rng.integers(
                0,
                len(cluster_names),
                size=len(cluster_names),
            )
            for key in COMPARISONS:
                cluster_values = []
                for cluster_index in sampled_clusters:
                    model_values = [
                        float(
                            row["_group_values"][key][
                                sampled_groups
                            ].mean()
                        )
                        for row in by_cluster[
                            cluster_names[int(cluster_index)]
                        ]
                    ]
                    cluster_values.append(float(np.mean(model_values)))
                dataset_draws[key].append(float(np.mean(cluster_values)))
        for key in COMPARISONS:
            outputs[key].append(
                float(np.mean(dataset_draws[key]))
            )
    return {
        key: np.asarray(values, dtype=float)
        for key, values in outputs.items()
    }


def _interval(values: np.ndarray) -> list[float]:
    return [
        float(value)
        for value in np.percentile(values, [2.5, 97.5])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    by_dataset: dict[str, list[dict]] = {
        dataset: [] for dataset in DATASETS
    }
    for spec in csecf._candidate_paths():
        states = csecf._load_checkpoint(spec)
        by_dataset[spec.dataset].append(
            _checkpoint_record(
                spec.dataset,
                spec.name,
                spec.run_cluster,
                states,
                rng,
            )
        )

    primary_draws = _hierarchical_draws(
        by_dataset,
        draws=AGGREGATE_DRAWS,
        seed=SEED,
    )
    stability_ranges = {key: [] for key in COMPARISONS}
    for offset in range(STABILITY_RUNS):
        stability_draws = _hierarchical_draws(
            by_dataset,
            draws=STABILITY_DRAWS,
            seed=5000 + offset,
        )
        for key in COMPARISONS:
            stability_ranges[key].append(
                _interval(stability_draws[key])
            )
    stability = {}
    for key in COMPARISONS:
        endpoint_ranges = np.asarray(
            stability_ranges[key],
            dtype=float,
        )
        stability[key] = [
            [
                float(endpoint_ranges[:, 0].min()),
                float(endpoint_ranges[:, 0].max()),
            ],
            [
                float(endpoint_ranges[:, 1].min()),
                float(endpoint_ranges[:, 1].max()),
            ],
        ]

    dataset_summary = {}
    for dataset_index, dataset in enumerate(DATASETS):
        rows = by_dataset[dataset]
        dataset_bootstraps = _dataset_draws(
            rows,
            draws=AGGREGATE_DRAWS,
            seed=SEED + 1 + dataset_index,
        )
        dataset_summary[dataset] = {
            "n_checkpoints": len(rows),
            "n_run_clusters": len(
                {row["run_cluster"] for row in rows}
            ),
            "comparisons": {
                key: {
                    "point_equal_group": _dataset_point(rows, key),
                    "run_cluster_by_group_ci95": _interval(
                        dataset_bootstraps[key]
                    ),
                }
                for key in COMPARISONS
            },
        }

    balanced = {}
    for key in COMPARISONS:
        point = float(
            np.mean([
                _dataset_point(by_dataset[dataset], key)
                for dataset in DATASETS
            ])
        )
        balanced[key] = {
            "point_equal_group": point,
            "run_cluster_by_group_ci95": _interval(
                primary_draws[key]
            ),
            "stability_ci95_endpoint_range": stability[key],
        }

    policy_summary = {}
    policy_names = (
        "visible",
        "cdf",
        "pia",
        "spg",
        "no_shield",
        "csecf",
        "cdf_only",
    )
    for policy in policy_names:
        per_dataset_gain = {
            dataset: _dataset_policy_point(
                by_dataset[dataset],
                policy,
            )
            for dataset in DATASETS
        }
        checkpoint_gains = {}
        for dataset in DATASETS:
            for row in by_dataset[dataset]:
                checkpoint_gains[f"{dataset}/{row['name']}"] = float(
                    row["_policy_group_gains"][policy].mean()
                )
        policy_summary[policy] = {
            "per_dataset_equal_group_gain": per_dataset_gain,
            "dataset_balanced_equal_group_gain": float(
                np.mean(list(per_dataset_gain.values()))
            ),
            "checkpoint_equal_group_gains": checkpoint_gains,
            "positive_checkpoints": int(
                sum(value > 0 for value in checkpoint_gains.values())
            ),
            "worst_checkpoint_gain": float(
                min(checkpoint_gains.values())
            ),
        }

    ap5095_sensitivity = {}
    ap5095_comparisons = {
        "csecf_minus_visible": ("csecf", "visible"),
        "csecf_minus_cdf": ("csecf", "cdf"),
        "cdf_only_minus_visible": ("cdf_only", "visible"),
        "csecf_minus_cdf_only": ("csecf", "cdf_only"),
        "no_shield_minus_visible": ("no_shield", "visible"),
    }
    checkpoint_ap5095 = {}
    for dataset in DATASETS:
        for row in by_dataset[dataset]:
            values = {
                key: (
                    row["policy_metrics"][left]["ap50_95"]
                    - row["policy_metrics"][right]["ap50_95"]
                )
                for key, (left, right) in ap5095_comparisons.items()
            }
            checkpoint_ap5095[f"{dataset}/{row['name']}"] = values
    for key in ap5095_comparisons:
        dataset_means = {
            dataset: float(np.mean([
                checkpoint_ap5095[f"{dataset}/{row['name']}"][key]
                for row in by_dataset[dataset]
            ]))
            for dataset in DATASETS
        }
        values = [
            row[key] for row in checkpoint_ap5095.values()
        ]
        ap5095_sensitivity[key] = {
            "description": (
                "Descriptive pooled-checkpoint AP50-95 difference; "
                "no interval or confirmatory inference."
            ),
            "per_dataset_mean": dataset_means,
            "dataset_balanced_mean": float(
                np.mean(list(dataset_means.values()))
            ),
            "checkpoint_values": {
                name: row[key]
                for name, row in checkpoint_ap5095.items()
            },
            "positive_checkpoints": int(
                sum(value > 0 for value in values)
            ),
            "worst_checkpoint": float(min(values)),
            "best_checkpoint": float(max(values)),
        }

    test_grid_equal_group = {}
    for key in by_dataset[DATASETS[0]][0][
        "policy_grid_equal_group"
    ]:
        dataset_means = {}
        for dataset in DATASETS:
            values = [
                row["policy_grid_equal_group"][key]["gain"]
                for row in by_dataset[dataset]
            ]
            dataset_means[dataset] = float(np.mean(values))
        checkpoint_values = {
            f"{dataset}/{row['name']}": row[
                "policy_grid_equal_group"
            ][key]["gain"]
            for dataset in DATASETS
            for row in by_dataset[dataset]
        }
        test_grid_equal_group[key] = {
            "dataset_balanced_equal_group_gain": float(
                np.mean(list(dataset_means.values()))
            ),
            "mean_checkpoint_gain": float(
                np.mean(list(checkpoint_values.values()))
            ),
            "worst_checkpoint_gain": float(
                min(checkpoint_values.values())
            ),
            "positive_checkpoints": int(
                sum(value > 0 for value in checkpoint_values.values())
            ),
            "mean_switch_fraction": float(np.mean([
                row["policy_grid_equal_group"][key][
                    "switch_fraction"
                ]
                for dataset in DATASETS
                for row in by_dataset[dataset]
            ])),
        }

    output = {
        "protocol": {
            "primary_estimand": (
                "paired group-level AP50 difference, with each evaluation "
                "group weighted equally within a dataset"
            ),
            "hierarchical_summary": (
                "checkpoints averaged within training-run cluster; LLVIP, "
                "M3FD, and DroneVehicle weighted equally"
            ),
            "group_units": {
                "llvip": "scene prefix",
                "m3fd": "prespecified held-out frame group",
                "dronevehicle_test": "prespecified 100-frame block",
            },
            "checkpoint_bootstrap_draws": CHECKPOINT_DRAWS,
            "aggregate_bootstrap_draws": AGGREGATE_DRAWS,
            "stability_runs": STABILITY_RUNS,
            "stability_draws_per_run": STABILITY_DRAWS,
            "seed": SEED,
            "point_and_interval_scope": (
                "The point, primary interval, and stability audit use the "
                "same equal-group estimand. Pooled AP50 and AP50-95 values "
                "are descriptive and are not the target of the interval."
            ),
        },
        "checkpoints": {
            f"{row['dataset']}/{row['name']}": {
                key: value
                for key, value in row.items()
                if not key.startswith("_")
            }
            for rows in by_dataset.values()
            for row in rows
        },
        "per_dataset": dataset_summary,
        "dataset_balanced": balanced,
        "policy_summary": policy_summary,
        "ap50_95_sensitivity": ap5095_sensitivity,
        "test_policy_grid_equal_group": test_grid_equal_group,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            dataset: {
                key: values["point_equal_group"]
                for key, values in dataset_summary[dataset][
                    "comparisons"
                ].items()
            }
            for dataset in dataset_summary
        },
        indent=2,
    ))
    print(json.dumps(
        {
            key: {
                "point": values["point_equal_group"],
                "ci95": values["run_cluster_by_group_ci95"],
            }
            for key, values in balanced.items()
        },
        indent=2,
    ))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
