"""Reanalyse CS-ECF for multiplicity, sensitivity, oracle, and random baselines.

The script starts from the released per-image detection scores. It does not run
detector inference. Set SSGF_DATA_ROOT when the data checkout differs from the
repository containing this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SSGF_DATA_ROOT", str(SCRIPT_ROOT)))
sys.path.insert(0, str(DATA_ROOT / "scripts"))

import analyze_csecf_vs_cdf as csecf  # noqa: E402
from local_ap_reference import load as local_load, pooled as local_pooled  # noqa: E402


TEST_POLICIES = (
    (1, 1.2),
    (1, 1.5),
    (1, 2.0),
    (3, 1.2),
    (3, 1.5),
    (3, 2.0),
    (5, 1.2),
    (5, 1.5),
    (5, 2.0),
    (10, 1.2),
    (10, 1.5),
    (10, 2.0),
)
MAIN_GRID = ((1, 1.2), (1, 1.5), (1, 2.0), (1, 2.5),
             (5, 1.2), (5, 1.5), (5, 2.0), (5, 2.5),
             (10, 1.2), (10, 1.5), (10, 2.0), (10, 2.5))


def _ap(stats: dict, indices: np.ndarray | None = None) -> float:
    if indices is None:
        indices = np.arange(stats["n"])
    return float(local_pooled(stats, np.asarray(indices, dtype=np.int64))[0])


def _score(data: dict, image_index: int, top_k: int) -> float:
    if "cf" in data:
        values = np.asarray(data["cf"][image_index], dtype=np.float64)
    else:
        mask = data["img_idx"] == image_index
        values = np.asarray(data["conf"][mask], dtype=np.float64)
    confidence = np.sort(values)[::-1]
    if len(confidence) == 0:
        return -1.0
    return float(confidence[:top_k].mean())


def _choices(states: dict[str, dict], top_k: int, margin: float) -> list[str]:
    choices = []
    for image in range(states["vis"]["n"]):
        values = {
            candidate: _score(state, image, top_k)
            for candidate, state in states.items()
        }
        best = max(("vis", "cdf", "pia", "spg"), key=lambda name: values[name])
        if best != "vis" and values[best] < margin * values["vis"]:
            best = "vis"
        choices.append(best)
    return choices


def _selected_stats(states: dict[str, dict], choices: list[str]) -> dict:
    return csecf._policy_stats(states, choices)


def _image_ap(state: dict, image_index: int) -> float:
    tp = np.asarray(state["tp"][image_index], dtype=float)
    confidence = np.asarray(state["cf"][image_index], dtype=float)
    n_gt = len(state["gt"][image_index])
    if n_gt == 0 or len(confidence) == 0:
        return 0.0
    order = np.argsort(-confidence)
    flags = tp[order, 0]
    true_positive = np.cumsum(flags)
    false_positive = np.cumsum(1.0 - flags)
    recall = true_positive / n_gt
    precision = true_positive / (true_positive + false_positive + 1e-12)
    return float(
        np.mean(
            [
                precision[recall >= point].max()
                if np.any(recall >= point)
                else 0.0
                for point in np.linspace(0.0, 1.0, 101)
            ]
        )
    )


def _two_sided_bootstrap_p(draws: np.ndarray) -> float:
    draws = np.asarray(draws, dtype=float)
    above = int(np.sum(draws >= 0.0))
    below = int(np.sum(draws <= 0.0))
    return float(
        min(
            1.0,
            2.0 * min(above + 1, below + 1) / (len(draws) + 1),
        )
    )


def _holm(p_values: list[float]) -> list[float]:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for position, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (count - position)))
        adjusted[index] = running
    return adjusted.tolist()


def _bh(p_values: list[float]) -> list[float]:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 1.0
    count = len(p_values)
    for reverse_position, index in enumerate(order[::-1], start=1):
        rank = count - reverse_position + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = running
    return adjusted.tolist()


def _checkpoint_task(arguments):
    checkpoint_index, spec, bootstrap_draws, random_draws = arguments
    states = csecf._load_checkpoint(spec)
    n_images = states["vis"]["n"]
    groups = csecf._groups(spec.dataset, states["vis"]["stems"])
    all_indices = np.arange(n_images)

    primary_choices = _choices(states, 5, 1.5)
    no_shield_choices = _choices(states, 5, 1.0)
    primary_stats = _selected_stats(states, primary_choices)
    no_shield_stats = _selected_stats(states, no_shield_choices)

    point = {
        "visible": _ap(states["vis"], all_indices),
        "always_cdf": _ap(states["cdf"], all_indices),
        "no_shield": _ap(no_shield_stats, all_indices),
        "csecf": _ap(primary_stats, all_indices),
    }

    random_rng = np.random.default_rng(8800 + checkpoint_index)
    random_aps = []
    candidate_names = ("vis", "cdf", "pia", "spg")
    for _ in range(random_draws):
        selected = random_rng.integers(0, len(candidate_names), size=n_images)
        choices = [candidate_names[index] for index in selected]
        random_aps.append(_ap(_selected_stats(states, choices), all_indices))
    random_aps = np.asarray(random_aps)

    image_aps = {
        candidate: np.asarray(
            [_image_ap(state, image) for image in range(n_images)]
        )
        for candidate, state in states.items()
    }
    oracle_choices = []
    for image in range(n_images):
        best = max(
            candidate_names,
            key=lambda name: (image_aps[name][image], name == "vis"),
        )
        oracle_choices.append(best)
    oracle_stats = _selected_stats(states, oracle_choices)
    oracle_ap = _ap(oracle_stats, all_indices)

    policy_grid = {}
    for top_k, margin in TEST_POLICIES:
        choices = _choices(states, top_k, margin)
        stats = _selected_stats(states, choices)
        policy_grid[f"K{top_k}_m{margin:.1f}"] = {
            "ap50": _ap(stats, all_indices),
            "gain": _ap(stats, all_indices) - point["visible"],
            "switch_fraction": (
                sum(choice != "vis" for choice in choices) / n_images
            ),
            "selected_counts": {
                candidate: choices.count(candidate)
                for candidate in candidate_names
            },
        }

    rng = np.random.default_rng(7100 + checkpoint_index)
    visible_draws = []
    cdf_draws = []
    for _ in range(bootstrap_draws):
        indices = csecf._draw_by_group(rng, groups)
        trial_ap = _ap(primary_stats, indices)
        visible_draws.append(trial_ap - _ap(states["vis"], indices))
        cdf_draws.append(trial_ap - _ap(states["cdf"], indices))
    visible_draws = np.asarray(visible_draws)
    cdf_draws = np.asarray(cdf_draws)

    group_visible = []
    group_cdf = []
    for indices in groups:
        trial_ap = _ap(primary_stats, indices)
        group_visible.append(trial_ap - _ap(states["vis"], indices))
        group_cdf.append(trial_ap - _ap(states["cdf"], indices))

    return {
        "dataset": spec.dataset,
        "name": spec.name,
        "run_cluster": spec.run_cluster,
        "n_images": n_images,
        "n_groups": len(groups),
        "point": point,
        "visible_delta": point["csecf"] - point["visible"],
        "cdf_delta": point["csecf"] - point["always_cdf"],
        "visible_group_ci95": [
            float(value) for value in np.percentile(visible_draws, [2.5, 97.5])
        ],
        "visible_p": _two_sided_bootstrap_p(visible_draws),
        "cdf_group_ci95": [
            float(value) for value in np.percentile(cdf_draws, [2.5, 97.5])
        ],
        "cdf_p": _two_sided_bootstrap_p(cdf_draws),
        "group_visible_gains": [float(value) for value in group_visible],
        "group_cdf_gains": [float(value) for value in group_cdf],
        "selected_counts": {
            candidate: primary_choices.count(candidate)
            for candidate in candidate_names
        },
        "no_shield_counts": {
            candidate: no_shield_choices.count(candidate)
            for candidate in candidate_names
        },
        "random": {
            "draws": random_draws,
            "mean_ap50": float(random_aps.mean()),
            "mean_gain": float(random_aps.mean() - point["visible"]),
            "ci95_gain": [
                float(value - point["visible"])
                for value in np.percentile(random_aps, [2.5, 97.5])
            ],
        },
        "oracle": {
            "method": "per-image AP50 maximum over vis/CDF/PIA/SPG; ties to visible",
            "ap50": oracle_ap,
            "gain": oracle_ap - point["visible"],
            "selected_counts": {
                candidate: oracle_choices.count(candidate)
                for candidate in candidate_names
            },
        },
        "policy_grid": policy_grid,
    }


def _dataset_hierarchical_draws(rows: list[dict], field: str,
                                draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clusters: dict[str, list[dict]] = {}
    for row in rows:
        clusters.setdefault(row["run_cluster"], []).append(row)
    cluster_names = sorted(clusters)
    output = []
    for _ in range(draws):
        sampled_clusters = rng.integers(
            0, len(cluster_names), size=len(cluster_names)
        )
        cluster_means = []
        for cluster_index in sampled_clusters:
            model_rows = clusters[cluster_names[int(cluster_index)]]
            model_means = []
            for row in model_rows:
                gains = np.asarray(row[field], dtype=float)
                sampled_groups = rng.integers(0, len(gains), size=len(gains))
                model_means.append(float(gains[sampled_groups].mean()))
            cluster_means.append(float(np.mean(model_means)))
        output.append(float(np.mean(cluster_means)))
    return np.asarray(output)


def _balanced_hierarchical_draws(rows: list[dict], field: str,
                                 draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    datasets = ("llvip", "m3fd", "dronevehicle_test")
    by_dataset = {
        dataset: [row for row in rows if row["dataset"] == dataset]
        for dataset in datasets
    }
    clusters = {
        dataset: {
            cluster: [
                row for row in dataset_rows
                if row["run_cluster"] == cluster
            ]
            for cluster in sorted(
                set(row["run_cluster"] for row in dataset_rows)
            )
        }
        for dataset, dataset_rows in by_dataset.items()
    }
    output = []
    for _ in range(draws):
        dataset_values = []
        for dataset in datasets:
            cluster_names = sorted(clusters[dataset])
            picked = rng.integers(0, len(cluster_names), size=len(cluster_names))
            cluster_values = []
            for cluster_index in picked:
                model_rows = clusters[dataset][cluster_names[int(cluster_index)]]
                model_values = []
                for row in model_rows:
                    gains = np.asarray(row[field], dtype=float)
                    sampled = rng.integers(0, len(gains), size=len(gains))
                    model_values.append(float(gains[sampled].mean()))
                cluster_values.append(float(np.mean(model_values)))
            dataset_values.append(float(np.mean(cluster_values)))
        output.append(float(np.mean(dataset_values)))
    return np.asarray(output)


def _validation_grid(data_root: Path) -> dict:
    results = data_root / "results" / "selective_fusion"
    profiles = {
        "llvip": {
            "models": ["yolo26n", "rtdetr"],
            "cdf": results / "validation_scores_pixelmean_cdf",
            "sota": results / "validation_scores_with_sota",
        },
        "m3fd": {
            "models": ["yolo26_m3fd6", "yolo8n_m3fd6"],
            "cdf": results / "m3fd_tau_validation_6cls",
            "sota": results / "m3fd_sota_validation",
        },
    }
    cdf_name = "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"
    output = {}
    for dataset, spec in profiles.items():
        output[dataset] = {}
        for model in spec["models"]:
            states = {
                "vis": local_load(spec["cdf"] / f"{model}__vis.npz"),
                "cdf": local_load(
                    spec["cdf"] / f"{model}__{cdf_name}.npz"
                ),
                "pia": local_load(
                    spec["sota"] / f"{model}__piafusion.npz"
                ),
                "spg": local_load(
                    spec["sota"] / f"{model}__spgfusion.npz"
                ),
            }
            n_images = int(states["vis"]["n"])
            baseline = _ap(states["vis"])
            entries = {}
            for top_k, margin in MAIN_GRID:
                choices = _choices(states, top_k, margin)
                selected = _selected_stats(states, choices)
                entries[f"K{top_k}_m{margin:.1f}"] = {
                    "gain": _ap(selected) - baseline,
                    "switch_fraction": (
                        sum(choice != "vis" for choice in choices) / n_images
                    ),
                }
            output[dataset][model] = entries
    return output


def _summarise_validation(grid: dict) -> dict:
    summaries = {}
    for key in next(iter(next(iter(grid.values())).values())):
        gains = []
        switches = []
        dataset_means = []
        for dataset in grid.values():
            profile_gains = []
            for model in dataset.values():
                gains.append(model[key]["gain"])
                switches.append(model[key]["switch_fraction"])
                profile_gains.append(model[key]["gain"])
            dataset_means.append(float(np.mean(profile_gains)))
        summaries[key] = {
            "dataset_balanced_gain": float(np.mean(dataset_means)),
            "mean_checkpoint_gain": float(np.mean(gains)),
            "min_gain": float(np.min(gains)),
            "positive_models": int(np.sum(np.asarray(gains) > 0)),
            "mean_switch_fraction": float(np.mean(switches)),
        }
    return summaries


def _summarise_test(rows: list[dict]) -> dict:
    output = {}
    for key in next(iter(rows))["policy_grid"]:
        gains = [row["policy_grid"][key]["gain"] for row in rows]
        switches = [
            row["policy_grid"][key]["switch_fraction"] for row in rows
        ]
        dataset_means = []
        for dataset in ("llvip", "m3fd", "dronevehicle_test"):
            dataset_gains = [
                row["policy_grid"][key]["gain"]
                for row in rows if row["dataset"] == dataset
            ]
            dataset_means.append(float(np.mean(dataset_gains)))
        output[key] = {
            "dataset_balanced_mean": float(np.mean(dataset_means)),
            "mean_checkpoint_gain": float(np.mean(gains)),
            "min_checkpoint_gain": float(np.min(gains)),
            "positive_checkpoints": int(np.sum(np.asarray(gains) > 0)),
            "mean_switch_fraction": float(np.mean(switches)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--random-draws", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    global csecf
    sys.path.insert(0, str(data_root / "scripts"))
    import analyze_csecf_vs_cdf as imported_csecf
    csecf = imported_csecf

    checkpoints = csecf._candidate_paths()
    tasks = [
        (index, checkpoint, args.bootstrap_draws, args.random_draws)
        for index, checkpoint in enumerate(checkpoints)
    ]
    if args.jobs == 1:
        rows = [_checkpoint_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(_checkpoint_task, tasks))

    names = [f"{row['dataset']}/{row['name']}" for row in rows]
    visible_ps = [row["visible_p"] for row in rows]
    visible_holm = _holm(visible_ps)
    visible_bh = _bh(visible_ps)
    for index, row in enumerate(rows):
        row["visible_p_holm"] = visible_holm[index]
        row["visible_q_bh"] = visible_bh[index]

    successes = sum(row["visible_delta"] > 0 for row in rows)
    sign_test_p = sum(
        math.comb(len(rows), count)
        for count in range(successes, len(rows) + 1)
    ) / (2 ** len(rows))

    dataset_names = ("llvip", "m3fd", "dronevehicle_test")
    paired_path = data_root / "results" / "selective_fusion" / "csecf_vs_cdf_paired.json"
    paired = json.loads(paired_path.read_text(encoding="utf-8"))

    stability = []
    for seed in range(20):
        draws = _balanced_hierarchical_draws(
            rows, "group_visible_gains", draws=5000, seed=5000 + seed
        )
        stability.append(np.percentile(draws, [2.5, 97.5]))
    stability = np.asarray(stability)

    validation_grid = _validation_grid(data_root)
    output = {
        "protocol": {
            "independent_unit": (
                "detector checkpoint for architecture comparisons; scene or "
                "consecutive frame group within checkpoint for paired AP50 intervals"
            ),
            "primary_family": (
                "12 prespecified CS-ECF-minus-visible checkpoint contrasts"
            ),
        "secondary_endpoint": (
            "single dataset-balanced CS-ECF-minus-CDF contrast; per-dataset "
            "breakdowns are descriptive"
        ),
            "multiplicity": "Holm family-wise correction; BH q values also reported",
            "checkpoint_bootstrap_draws": args.bootstrap_draws,
            "aggregate_bootstrap_draws": 10000,
            "aggregate_stability_runs": 20,
            "aggregate_stability_draws_per_run": 5000,
            "random_draws_per_checkpoint": args.random_draws,
            "random_seeds": {
                "checkpoint_bootstraps": "7100 + checkpoint index",
                "random_baselines": "8800 + checkpoint index",
            },
            "oracle": (
                "candidate with maximum per-image AP50 selected with test labels; "
                "upper bound only"
            ),
        },
        "checkpoints": {name: row for name, row in zip(names, rows)},
        "multiple_comparisons": {
            "visible_relative": {
                "family_size": len(rows),
                "positive_checkpoints": successes,
                "exact_sign_test_p": sign_test_p,
                "survive_holm": [
                    name for name, q in zip(names, visible_holm) if q < 0.05
                ],
                "survive_bh": [
                    name for name, q in zip(names, visible_bh) if q < 0.05
                ],
            },
        },
        "cdf_comparison": {
            "role": "single prespecified secondary estimand; per-dataset descriptive",
            "dataset_balanced": paired["dataset_balanced"],
            "per_dataset": paired["per_dataset"],
        },
        "dataset_balanced": {
            "visible": {
                "point": paired["dataset_balanced"]["point_csecf_minus_visible"],
                "hierarchical_ci95": paired["dataset_balanced"][
                    "visible_run_cluster_by_group_ci95"
                ],
                "stability_ci95_endpoint_range": [
                    [float(stability[:, 0].min()), float(stability[:, 0].max())],
                    [float(stability[:, 1].min()), float(stability[:, 1].max())],
                ],
            },
            "cdf": {
                "point": paired["dataset_balanced"][
                    "point_csecf_minus_cdf"
                ],
                "hierarchical_ci95": paired["dataset_balanced"][
                    "run_cluster_by_group_ci95"
                ],
            },
        },
        "validation_policy_grid": {
            "grid": validation_grid,
            "summary": _summarise_validation(validation_grid),
            "selection_note": (
                "The four-candidate top-five rule was fixed before margin "
                "selection. Among the pre-specified margin grid, 1.5 was the "
                "first setting satisfying the 0.0005 per-evaluator loss bound "
                "and the largest balanced validation gain under that rule."
            ),
        },
        "test_policy_grid": _summarise_test(rows),
        "baselines": {
            "random": {
                "dataset_balanced_gain": float(np.mean([
                    np.mean([
                        row["random"]["mean_gain"]
                        for row in rows if row["dataset"] == dataset
                    ])
                    for dataset in dataset_names
                ])),
                "mean_checkpoint_gain": float(np.mean([
                    row["random"]["mean_gain"] for row in rows
                ])),
                "min_checkpoint_gain": float(
                    np.min([row["random"]["mean_gain"] for row in rows])
                ),
                "max_checkpoint_gain": float(
                    np.max([row["random"]["mean_gain"] for row in rows])
                ),
            },
            "oracle": {
                "dataset_balanced_gain": float(np.mean([
                    np.mean([
                        row["oracle"]["gain"]
                        for row in rows if row["dataset"] == dataset
                    ])
                    for dataset in dataset_names
                ])),
                "mean_checkpoint_gain": float(np.mean([
                    row["oracle"]["gain"] for row in rows
                ])),
                "min_checkpoint_gain": float(
                    np.min([row["oracle"]["gain"] for row in rows])
                ),
                "max_checkpoint_gain": float(
                    np.max([row["oracle"]["gain"] for row in rows])
                ),
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "sign_test_p": sign_test_p,
        "visible_survive_holm": output["multiple_comparisons"][
            "visible_relative"
        ]["survive_holm"],
        "visible_survive_bh": output["multiple_comparisons"][
            "visible_relative"
        ]["survive_bh"],
        "dataset_balanced_visible": output["dataset_balanced"]["visible"],
        "random": output["baselines"]["random"],
        "oracle": output["baselines"]["oracle"],
    }, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
