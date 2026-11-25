"""Evaluate cost-aware candidate-library selection for CS-ECF.

The analysis keeps the frozen score depth and visible margin unchanged. It
constructs every candidate subset that contains visible, evaluates each subset
on the original four development evaluator cells, and selects the smallest
subset that is practically equivalent to the best feasible validation subset.

The selected subset is then evaluated without modification on:

- the confidence-ordered LLVIP holdout in the submitted evidence package;
- standard-matching VEDAI;
- the two independent DroneVehicle runs;
- the retrospective 12-checkpoint maximum-cardinality score matrix.

This is a post-review analysis. The candidate-selection rule is fixed in this
script, but the new subset was not part of the original prospective protocol.
The output therefore labels the result as exploratory rather than confirmatory.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    path
    for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
    if (path / "scripts" / "prospective_llvip_analysis.py").is_file()
)
SCRIPTS = ROOT / "scripts"
ANALYSIS = ROOT / "analysis" / "cs_ecf"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ANALYSIS))

import analyze_csecf_vs_cdf as retrospective  # noqa: E402
import major_revision_statistics as revision  # noqa: E402
import prospective_llvip_analysis as prospective  # noqa: E402
from local_ap_reference import load, pooled  # noqa: E402


CANDIDATES = ("vis", "cdf", "pia", "spg")
FUSED_CANDIDATES = ("cdf", "pia", "spg")
TOP_K = 5
FROZEN_MARGIN = 1.5
MARGINS = (1.2, 1.3, 1.5, 1.75, 2.0)
VALIDATION_LOSS_BOUND = 0.0005
PRACTICAL_EQUIVALENCE = 1e-5

DETECTOR_ONCE_MS = 32.5
FUSION_MS = {
    "cdf": 22.282,
    "pia": 37.7,
    "spg": 516.3,
}
FUSION_PEAK_MIB = {
    "cdf": None,
    "pia": 869.1,
    "spg": 1453.5,
}

CDF_NAME = "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"
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
DEFAULT_OUTPUT = (
    DEFAULT_PACKAGE / "evidence" / "budgeted_selection"
    if SCRIPT_PATH.parent.name == "reproduction"
    else ANALYSIS / "results" / "budgeted_selection"
)


@dataclass(frozen=True)
class Policy:
    candidates: tuple[str, ...]
    margin: float
    top_k: int = TOP_K

    @property
    def name(self) -> str:
        return "+".join(self.candidates) + f"@m{self.margin:g}"

    @property
    def detector_forwards(self) -> int:
        return len(self.candidates)

    @property
    def fusion_latency_ms(self) -> float:
        return float(sum(
            FUSION_MS[candidate]
            for candidate in self.candidates
            if candidate != "vis"
        ))

    @property
    def estimated_serial_ms(self) -> float:
        return (
            self.detector_forwards * DETECTOR_ONCE_MS
            + self.fusion_latency_ms
        )

    @property
    def max_independent_peak_mib(self) -> float | None:
        values = [
            FUSION_PEAK_MIB[candidate]
            for candidate in self.candidates
            if candidate != "vis"
            and FUSION_PEAK_MIB[candidate] is not None
        ]
        return float(max(values)) if values else None

    def as_payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "name": self.name,
            "detector_forwards": self.detector_forwards,
            "fusion_latency_ms": round(self.fusion_latency_ms, 3),
            "estimated_serial_ms": round(self.estimated_serial_ms, 3),
            "max_independent_peak_mib": self.max_independent_peak_mib,
        }


def candidate_subsets() -> tuple[tuple[str, ...], ...]:
    subsets: list[tuple[str, ...]] = [("vis",)]
    for size in range(1, len(FUSED_CANDIDATES) + 1):
        for fused in itertools.combinations(FUSED_CANDIDATES, size):
            subsets.append(("vis",) + fused)
    return tuple(subsets)


def topk_score(values: np.ndarray, top_k: int = TOP_K) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return -1.0
    return float(np.sort(values)[-top_k:].mean())


def candidate_choices(
    states: dict[str, dict[str, Any]],
    policy: Policy,
) -> list[str]:
    if len(policy.candidates) == 1:
        return [policy.candidates[0]] * int(states["vis"]["n"])
    choices: list[str] = []
    for image in range(int(states["vis"]["n"])):
        scores = {
            candidate: topk_score(
                states[candidate]["cf"][image],
                policy.top_k,
            )
            for candidate in policy.candidates
        }
        best = max(policy.candidates, key=lambda name: scores[name])
        if best != "vis" and scores[best] >= policy.margin * scores["vis"]:
            choices.append(best)
        else:
            choices.append("vis")
    return choices


def selected_stats(
    states: dict[str, dict[str, Any]],
    choices: list[str],
) -> dict[str, Any]:
    return revision._selected_stats(states, choices)


def pooled_ap50(stats: dict[str, Any], indices: np.ndarray) -> float:
    return float(pooled(stats, np.asarray(indices, dtype=np.int64))[0])


def pooled_ap50_95(stats: dict[str, Any], indices: np.ndarray) -> float:
    return float(pooled(stats, np.asarray(indices, dtype=np.int64))[1])


def load_validation_states() -> dict[str, dict[str, dict[str, Any]]]:
    results = ROOT / "results" / "selective_fusion"
    profiles = (
        {
            "dataset": "llvip",
            "models": ("yolo26n", "rtdetr"),
            "cdf_dir": results / "validation_scores_pixelmean_cdf",
            "sota_dir": results / "validation_scores_with_sota",
        },
        {
            "dataset": "m3fd",
            "models": ("yolo26_m3fd6", "yolo8n_m3fd6"),
            "cdf_dir": results / "m3fd_tau_validation_6cls",
            "sota_dir": results / "m3fd_sota_validation",
        },
    )
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for profile in profiles:
        for model in profile["models"]:
            key = f"{profile['dataset']}/{model}"
            output[key] = {
                "vis": load(profile["cdf_dir"] / f"{model}__vis.npz"),
                "cdf": load(
                    profile["cdf_dir"] / f"{model}__{CDF_NAME}.npz"
                ),
                "pia": load(
                    profile["sota_dir"] / f"{model}__piafusion.npz"
                ),
                "spg": load(
                    profile["sota_dir"] / f"{model}__spgfusion.npz"
                ),
            }
    return output


def evaluate_validation() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    states_by_cell = load_validation_states()
    rows: list[dict[str, Any]] = []
    for policy_candidates in candidate_subsets():
        for margin in MARGINS:
            policy = Policy(policy_candidates, margin)
            cell_gains: dict[str, float] = {}
            switch_fractions: dict[str, float] = {}
            for cell, states in states_by_cell.items():
                indices = np.arange(int(states["vis"]["n"]))
                choices = candidate_choices(states, policy)
                stats = selected_stats(states, choices)
                cell_gains[cell] = (
                    pooled_ap50(stats, indices)
                    - pooled_ap50(states["vis"], indices)
                )
                switch_fractions[cell] = float(np.mean([
                    choice != "vis" for choice in choices
                ]))
            gains = list(cell_gains.values())
            rows.append({
                "policy": policy.name,
                "candidates": list(policy.candidates),
                "margin": policy.margin,
                "mean_gain": float(np.mean(gains)),
                "worst_cell_gain": float(np.min(gains)),
                "positive_cells": int(sum(value > 0 for value in gains)),
                "mean_switch_fraction": float(
                    np.mean(list(switch_fractions.values()))
                ),
                "cell_gains": cell_gains,
                "cell_switch_fractions": switch_fractions,
                "feasible": bool(
                    np.min(gains) >= -VALIDATION_LOSS_BOUND
                ),
                **policy.as_payload(),
            })

    frozen_rows = [
        row for row in rows
        if abs(float(row["margin"]) - FROZEN_MARGIN) < 1e-12
    ]
    feasible = [row for row in frozen_rows if row["feasible"]]
    if not feasible:
        raise RuntimeError("No feasible candidate subset at frozen margin")
    best_mean = max(float(row["mean_gain"]) for row in feasible)
    shortlist = [
        row for row in feasible
        if best_mean - float(row["mean_gain"]) <= PRACTICAL_EQUIVALENCE
    ]
    shortlist.sort(key=lambda row: (
        int(row["detector_forwards"]),
        float(row["estimated_serial_ms"]),
        -float(row["mean_gain"]),
        "-".join(row["candidates"]),
    ))
    selected = shortlist[0]
    selection = {
        "procedure": (
            "Fix K=5 and margin=1.5. For every subset containing visible, "
            "require every development evaluator gain to be at least "
            "-0.0005 AP50. Among feasible subsets, take those within 0.00001 "
            "of the largest mean gain, then choose the fewest detector "
            "forwards and lowest measured component cost."
        ),
        "frozen_margin": FROZEN_MARGIN,
        "validation_loss_bound": -VALIDATION_LOSS_BOUND,
        "practical_equivalence_gain": PRACTICAL_EQUIVALENCE,
        "selected_policy": selected["policy"],
        "selected_candidates": selected["candidates"],
        "selected_validation_mean_gain": selected["mean_gain"],
        "selected_validation_worst_cell_gain": selected[
            "worst_cell_gain"
        ],
        "selected_validation_positive_cells": selected["positive_cells"],
        "selected_cost": {
            "detector_forwards": selected["detector_forwards"],
            "fusion_latency_ms": selected["fusion_latency_ms"],
            "estimated_serial_ms": selected["estimated_serial_ms"],
            "max_independent_peak_mib": selected[
                "max_independent_peak_mib"
            ],
        },
        "shortlist": [row["policy"] for row in shortlist],
        "status": "exploratory_post_review_analysis",
    }
    return rows, selection


def policy_for_payload(policy: str | Policy, margin: float | None = None) -> Policy:
    if isinstance(policy, Policy):
        return policy
    candidates = (
        tuple(policy.split("+"))
        if isinstance(policy, str)
        else tuple(policy)
    )
    if not candidates or candidates[0] != "vis":
        raise ValueError(f"invalid policy: {policy}")
    if any(candidate not in CANDIDATES for candidate in candidates):
        raise ValueError(f"invalid policy: {policy}")
    return Policy(candidates, FROZEN_MARGIN if margin is None else margin)


def summarize_records(
    records: dict[str, dict[str, dict[str, Any]]],
    models: Iterable[str],
    sequences: Iterable[str],
) -> dict[str, Any]:
    models = tuple(models)
    sequences = tuple(sequences)
    per_checkpoint = {
        model: float(np.mean([
            records[model][sequence]["point"] for sequence in sequences
        ]))
        for model in models
    }
    per_sequence = {
        sequence: float(np.mean([
            records[model][sequence]["point"] for model in models
        ]))
        for sequence in sequences
    }
    values = list(per_checkpoint.values())
    return {
        "mean": float(np.mean(values)),
        "positive_checkpoints": int(sum(value > 0 for value in values)),
        "worst_checkpoint": float(min(values)),
        "per_checkpoint": per_checkpoint,
        "per_sequence": per_sequence,
    }


def evaluate_llvip_standard(
    score_root: Path,
    data_root: Path,
    selected_policy: Policy,
) -> dict[str, Any]:
    sequences = prospective.CONFIRM_SEQUENCES
    models = prospective.MODEL_NAMES
    policies = {
        "visible": Policy(("vis",), 1.0),
        "always_cdf": Policy(("cdf",), 1.0),
        "candidate": selected_policy,
        "full_csecf": Policy(CANDIDATES, FROZEN_MARGIN),
        "no_shield": Policy(CANDIDATES, 0.0),
    }
    policy_stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    blocks_by_cell: dict[tuple[str, str], list[np.ndarray]] = {}
    choices_by_cell: dict[tuple[str, str, str], list[str]] = {}

    for sequence in sequences:
        manifest_path = (
            data_root / f"erf_holdout_llvip{sequence}" / "manifest.json"
        )
        stems = json.loads(manifest_path.read_text(encoding="utf-8"))["stems"]
        score_dir = score_root / f"llvip{sequence}" / "scores"
        blocks = prospective._blocks(stems)
        for model in models:
            states = prospective._load_model(score_dir, model, stems)
            blocks_by_cell[(sequence, model)] = blocks
            for name, policy in policies.items():
                choices = candidate_choices(states, policy)
                choices_by_cell[(sequence, model, name)] = choices
                policy_stats[(sequence, model, name)] = selected_stats(
                    states,
                    choices,
                )

    comparisons: dict[str, Any] = {}
    for left, right in (
        ("candidate", "visible"),
        ("candidate", "no_shield"),
        ("candidate", "full_csecf"),
        ("full_csecf", "visible"),
        ("full_csecf", "always_cdf"),
        ("full_csecf", "no_shield"),
        ("no_shield", "visible"),
    ):
        records: dict[str, dict[str, dict[str, Any]]] = {
            model: {} for model in models
        }
        for model in models:
            for sequence in sequences:
                blocks = blocks_by_cell[(sequence, model)]
                block_gains = np.asarray([
                    pooled_ap50(
                        policy_stats[(sequence, model, left)],
                        block,
                    )
                    - pooled_ap50(
                        policy_stats[(sequence, model, right)],
                        block,
                    )
                    for block in blocks
                ])
                records[model][sequence] = {
                    "point": float(block_gains.mean()),
                    "block_gains": block_gains,
                }
        comparisons[f"{left}_minus_{right}"] = {
            **summarize_records(records, models, sequences),
            "conditional_sequence_by_run_interval": (
                prospective._bootstrap_interval(records)
            ),
        }

        records_ap95: dict[str, dict[str, dict[str, Any]]] = {
            model: {} for model in models
        }
        for model in models:
            for sequence in sequences:
                blocks = blocks_by_cell[(sequence, model)]
                block_gains = np.asarray([
                    pooled_ap50_95(
                        policy_stats[(sequence, model, left)],
                        block,
                    )
                    - pooled_ap50_95(
                        policy_stats[(sequence, model, right)],
                        block,
                    )
                    for block in blocks
                ])
                records_ap95[model][sequence] = {
                    "point": float(block_gains.mean()),
                    "block_gains": block_gains,
                }
        comparisons[f"{left}_minus_{right}"]["ap50_95"] = {
            **summarize_records(records_ap95, models, sequences),
            "conditional_sequence_by_run_interval": (
                prospective._bootstrap_interval(records_ap95)
            ),
        }

    switch_fractions = []
    selected_counts = {candidate: 0 for candidate in CANDIDATES}
    for model in models:
        for sequence in sequences:
            choices = choices_by_cell[(sequence, model, "candidate")]
            switch_fractions.append(float(np.mean([
                choice != "vis" for choice in choices
            ])))
            for choice in choices:
                selected_counts[choice] += 1
    comparisons["candidate_minus_visible"]["mean_switch_fraction"] = float(
        np.mean(switch_fractions)
    )
    comparisons["candidate_minus_visible"]["selected_counts"] = selected_counts
    return {
        "matching": "confidence-ordered one-to-one matching",
        "sequences": list(sequences),
        "models": list(models),
        "policies": {
            name: policy.as_payload() for name, policy in policies.items()
        },
        "comparisons": comparisons,
        "status": "exploratory_post_review_analysis",
    }


def bootstrap_run_by_block(
    per_run: dict[str, np.ndarray],
    draws: int = 10000,
    seed: int = 6401,
) -> list[float]:
    rng = np.random.default_rng(seed)
    names = tuple(per_run)
    values = []
    for _ in range(draws):
        sampled_runs = rng.choice(names, size=len(names), replace=True)
        run_means = []
        for run in sampled_runs:
            gains = np.asarray(per_run[str(run)], dtype=np.float64)
            sampled_blocks = rng.integers(
                0,
                len(gains),
                size=len(gains),
            )
            run_means.append(float(gains[sampled_blocks].mean()))
        values.append(float(np.mean(run_means)))
    return [
        float(value)
        for value in np.percentile(values, [2.5, 97.5])
    ]


def evaluate_run_collection(
    runs: dict[str, dict[str, dict[str, Any]]],
    blocks_by_run: dict[str, list[np.ndarray]],
    selected_policy: Policy,
) -> dict[str, Any]:
    policies = {
        "visible": Policy(("vis",), 1.0),
        "always_cdf": Policy(("cdf",), 1.0),
        "candidate": selected_policy,
        "full_csecf": Policy(CANDIDATES, FROZEN_MARGIN),
        "no_shield": Policy(CANDIDATES, 0.0),
    }
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    choices: dict[tuple[str, str], list[str]] = {}
    for run, states in runs.items():
        for name, policy in policies.items():
            run_choices = candidate_choices(states, policy)
            choices[(run, name)] = run_choices
            stats[(run, name)] = selected_stats(states, run_choices)

    comparisons: dict[str, Any] = {}
    for left, right in (
        ("candidate", "visible"),
        ("candidate", "no_shield"),
        ("candidate", "full_csecf"),
        ("full_csecf", "visible"),
        ("full_csecf", "always_cdf"),
        ("full_csecf", "no_shield"),
        ("no_shield", "visible"),
    ):
        per_run: dict[str, np.ndarray] = {}
        for run in runs:
            block_gains = np.asarray([
                pooled_ap50(stats[(run, left)], block)
                - pooled_ap50(stats[(run, right)], block)
                for block in blocks_by_run[run]
            ])
            per_run[run] = block_gains
        run_means = {
            run: float(values.mean())
            for run, values in per_run.items()
        }
        comparisons[f"{left}_minus_{right}"] = {
            "equal_run_mean": float(np.mean(list(run_means.values()))),
            "per_run": run_means,
            "positive_runs": int(sum(value > 0 for value in run_means.values())),
            "worst_run": float(min(run_means.values())),
            "conditional_run_by_block_interval": bootstrap_run_by_block(
                per_run
            ),
        }

    switch_fractions = {
        run: float(np.mean([
            choice != "vis" for choice in choices[(run, "candidate")]
        ]))
        for run in runs
    }
    selected_counts = {
        run: {
            candidate: int(sum(
                choice == candidate
                for choice in choices[(run, "candidate")]
            ))
            for candidate in CANDIDATES
        }
        for run in runs
    }
    return {
        "runs": list(runs),
        "policies": {
            name: policy.as_payload() for name, policy in policies.items()
        },
        "comparisons": comparisons,
        "switch_fraction": switch_fractions,
        "selected_counts": selected_counts,
        "status": "exploratory_post_review_analysis",
    }


def load_vedai_runs(score_root: Path) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, list[np.ndarray]],
]:
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    blocks: dict[str, list[np.ndarray]] = {}
    for run in ("vedai_run1", "vedai_run2"):
        states = {
            candidate: load(score_root / f"{run}__{candidate}.npz")
            for candidate in CANDIDATES
        }
        runs[run] = states
        blocks[run] = [
            np.asarray(block, dtype=np.int64)
            for block in np.array_split(
                np.arange(int(states["vis"]["n"])),
                10,
            )
        ]
    return runs, blocks


def drone_blocks(stems: list[str]) -> list[np.ndarray]:
    labels = []
    for stem in stems:
        match = re.search(r"test_(\d+)", stem)
        if match is None:
            raise ValueError(f"unrecognized DroneVehicle stem: {stem}")
        labels.append((int(match.group(1)) - 1) // 100)
    labels_array = np.asarray(labels, dtype=np.int64)
    return [
        np.flatnonzero(labels_array == label)
        for label in sorted(set(labels))
    ]


def load_drone_runs(score_root: Path) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, list[np.ndarray]],
]:
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    blocks: dict[str, list[np.ndarray]] = {}
    for run in ("yolo11n_obb_run1", "yolo11n_obb_run2"):
        states = {
            candidate: load(score_root / f"{run}__{candidate}.npz")
            for candidate in CANDIDATES
        }
        runs[run] = states
        blocks[run] = drone_blocks(states["vis"]["stems"])
    return runs, blocks


def load_external_runs(score_root: Path) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, list[np.ndarray]],
]:
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    blocks: dict[str, list[np.ndarray]] = {}
    file_stems = {
        "vis": "vis",
        "cdf": "cdf",
        "pia": "piafusion",
        "spg": "spgfusion",
    }
    for model in ("yolov9t_coco", "yolov10s_coco", "yolov9t_domain"):
        model_dir = (
            score_root / "per_image"
            if model != "yolov9t_domain"
            else score_root / "domain_yolov9t_v3" / "per_image"
        )
        states = {
            candidate: load(
                model_dir / f"{model}__{file_stems[candidate]}.npz"
            )
            for candidate in CANDIDATES
        }
        runs[model] = states
        labels = np.asarray([
            stem[:3] for stem in states["vis"]["stems"]
        ])
        blocks[model] = [
            np.flatnonzero(labels == label)
            for label in sorted(set(labels))
        ]
    return runs, blocks


def evaluate_retrospective_grid() -> list[dict[str, Any]]:
    checkpoints = retrospective._candidate_paths()
    rows: list[dict[str, Any]] = []
    for policy_candidates in candidate_subsets():
        for margin in MARGINS:
            policy = Policy(policy_candidates, margin)
            per_dataset: dict[str, list[float]] = {
                "llvip": [],
                "m3fd": [],
                "dronevehicle_test": [],
            }
            switch_fractions = []
            for spec in checkpoints:
                states = retrospective._load_checkpoint(spec)
                groups = retrospective._groups(
                    spec.dataset,
                    states["vis"]["stems"],
                )
                choices = candidate_choices(states, policy)
                stats = selected_stats(states, choices)
                checkpoint_gains = [
                    pooled_ap50(stats, group)
                    - pooled_ap50(states["vis"], group)
                    for group in groups
                ]
                per_dataset[spec.dataset].append(
                    float(np.mean(checkpoint_gains))
                )
                switch_fractions.append(float(np.mean([
                    choice != "vis" for choice in choices
                ])))
            dataset_means = {
                dataset: float(np.mean(values))
                for dataset, values in per_dataset.items()
            }
            gains = [
                value
                for values in per_dataset.values()
                for value in values
            ]
            rows.append({
                "policy": policy.name,
                "candidates": list(policy.candidates),
                "margin": policy.margin,
                "dataset_balanced_gain": float(
                    np.mean(list(dataset_means.values()))
                ),
                "mean_checkpoint_gain": float(np.mean(gains)),
                "worst_checkpoint_gain": float(np.min(gains)),
                "positive_checkpoints": int(sum(value > 0 for value in gains)),
                "mean_switch_fraction": float(np.mean(switch_fractions)),
                "dataset_means": dataset_means,
                "per_dataset_gains": per_dataset,
                **policy.as_payload(),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "policy",
        "candidates",
        "margin",
        "mean_gain",
        "worst_cell_gain",
        "positive_cells",
        "mean_switch_fraction",
        "feasible",
        "dataset_balanced_gain",
        "mean_checkpoint_gain",
        "worst_checkpoint_gain",
        "positive_checkpoints",
        "detector_forwards",
        "fusion_latency_ms",
        "estimated_serial_ms",
        "max_independent_peak_mib",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            cleaned = {
                key: (
                    "+".join(value)
                    if key == "candidates"
                    else value
                )
                for key, value in row.items()
                if key in fields
            }
            writer.writerow(cleaned)


def write_paper_source_data(
    path: Path,
    payload: dict[str, Any],
    validation_rows: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for validation in validation_rows:
        if abs(float(validation["margin"]) - FROZEN_MARGIN) >= 1e-12:
            continue
        rows.append({
            "section": "validation subset",
            "comparison": validation["policy"],
            "estimate": validation["mean_gain"],
            "interval_low": "",
            "interval_high": "",
            "positive_units": validation["positive_cells"],
            "worst_unit": validation["worst_cell_gain"],
            "detector_forwards": validation["detector_forwards"],
            "estimated_serial_ms": validation["estimated_serial_ms"],
        })
    for section, key in (
        ("LLVIP standard", "llvip_standard_matching"),
        ("VEDAI standard", "vedai_standard_matching"),
        ("DroneVehicle standard", "dronevehicle_standard_matching"),
        ("DroneVehicle native", "dronevehicle_native_obb"),
        ("External evaluators", "external_evaluators"),
    ):
        for comparison in (
            "candidate_minus_visible",
            "candidate_minus_no_shield",
            "candidate_minus_full_csecf",
            "full_csecf_minus_visible",
            "full_csecf_minus_always_cdf",
            "full_csecf_minus_no_shield",
            "no_shield_minus_visible",
        ):
            row = payload[key]["comparisons"][comparison]
            if "mean" in row:
                estimate = row["mean"]
                low, high = row["conditional_sequence_by_run_interval"]
                positive = row["positive_checkpoints"]
                worst = row["worst_checkpoint"]
            else:
                estimate = row["equal_run_mean"]
                low, high = row["conditional_run_by_block_interval"]
                positive = row["positive_runs"]
                worst = row["worst_run"]
            rows.append({
                "section": section,
                "comparison": comparison,
                "estimate": estimate,
                "interval_low": low,
                "interval_high": high,
                "positive_units": positive,
                "worst_unit": worst,
                "detector_forwards": "",
                "estimated_serial_ms": "",
            })
    prospective_path = (
        path.parents[1]
        / "evidence"
        / "prospective_pia_confirmation"
        / "analysis.json"
    )
    if prospective_path.is_file():
        prospective = json.loads(
            prospective_path.read_text(encoding="utf-8")
        )
        primary = prospective["primary"]
        rows.append({
            "section": "LLVIP prospective 080-100",
            "comparison": "candidate_minus_visible",
            "estimate": primary["point"],
            "interval_low": primary["conditional_interval"][0],
            "interval_high": primary["conditional_interval"][1],
            "positive_units": primary["positive_sequences"],
            "worst_unit": primary["worst_checkpoint"],
            "detector_forwards": "",
            "estimated_serial_ms": "",
        })
    fields = (
        "section",
        "comparison",
        "estimate",
        "interval_low",
        "interval_high",
        "positive_units",
        "worst_unit",
        "detector_forwards",
        "estimated_serial_ms",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=DEFAULT_PACKAGE,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
    )
    args = parser.parse_args()

    validation_rows, selection = evaluate_validation()
    selected_policy = policy_for_payload(
        selection["selected_candidates"],
        selection["frozen_margin"],
    )

    llvip = evaluate_llvip_standard(
        args.package / "evidence" / "prospective_standard_matching",
        args.data_root,
        selected_policy,
    )
    vedai_runs, vedai_blocks = load_vedai_runs(
        args.package / "evidence" / "vedai" / "standard_matching"
    )
    vedai = evaluate_run_collection(
        vedai_runs,
        vedai_blocks,
        selected_policy,
    )
    drone_runs, drone_blocks = load_drone_runs(
        ROOT / "results" / "prospective_20261029"
        / "dronevehicle" / "native_scores"
    )
    dronevehicle = evaluate_run_collection(
        drone_runs,
        drone_blocks,
        selected_policy,
    )
    drone_standard_root = next(
        (
            path
            for path in (
                DEFAULT_PACKAGE / "evidence"
                / "dronevehicle_standard_matching",
                ROOT / "results" / "prospective_20261029"
                / "dronevehicle" / "standard_matching_new_runs",
            )
            if path.is_dir()
        ),
        None,
    )
    if drone_standard_root is not None:
        drone_standard_runs, drone_standard_blocks = load_drone_runs(
            drone_standard_root
        )
        dronevehicle_standard = evaluate_run_collection(
            drone_standard_runs,
            drone_standard_blocks,
            selected_policy,
        )
    else:
        dronevehicle_standard = None
    external_runs, external_blocks = load_external_runs(
        ROOT / "results" / "independent_evaluators"
    )
    external_evaluators = evaluate_run_collection(
        external_runs,
        external_blocks,
        selected_policy,
    )
    retrospective_rows = evaluate_retrospective_grid()

    payload = {
        "protocol": {
            "task": "budgeted candidate-library selection",
            "top_k": TOP_K,
            "frozen_margin": FROZEN_MARGIN,
            "margin_sensitivity_grid": list(MARGINS),
            "cost_model": {
                "device": "NVIDIA GeForce RTX 2080 Ti",
                "detector_once_ms": DETECTOR_ONCE_MS,
                "fusion_ms": FUSION_MS,
                "note": (
                    "Serial component estimate with models already loaded. "
                    "Peak memory is reported as the largest independently "
                    "measured fusion peak and is not additive."
                ),
            },
            "status": "exploratory_post_review_analysis",
        },
        "selection": selection,
        "llvip_standard_matching": llvip,
        "vedai_standard_matching": vedai,
        "dronevehicle_native_obb": dronevehicle,
        "dronevehicle_standard_matching": dronevehicle_standard,
        "external_evaluators": external_evaluators,
        "retrospective_max_cardinality": {
            "checkpoint_count": len(retrospective._candidate_paths()),
            "margin_sensitivity_grid": list(MARGINS),
            "rows": retrospective_rows,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.out / "validation_grid.csv", validation_rows)
    write_csv(args.out / "retrospective_grid.csv", retrospective_rows)
    write_paper_source_data(
        args.out / "FigureBudgetedSelection_source.csv",
        payload,
        validation_rows,
    )
    print(json.dumps({
        "selected_policy": selection["selected_policy"],
        "validation": {
            "mean_gain": selection["selected_validation_mean_gain"],
            "worst_cell_gain": selection[
                "selected_validation_worst_cell_gain"
            ],
        },
        "llvip": {
            key: {
                "mean": value["mean"],
                "interval": value[
                    "conditional_sequence_by_run_interval"
                ],
                "worst_checkpoint": value["worst_checkpoint"],
            }
            for key, value in llvip["comparisons"].items()
        },
        "vedai": {
            key: {
                "mean": value["equal_run_mean"],
                "interval": value["conditional_run_by_block_interval"],
            }
            for key, value in vedai["comparisons"].items()
        },
        "dronevehicle": {
            key: {
                "mean": value["equal_run_mean"],
                "interval": value["conditional_run_by_block_interval"],
            }
            for key, value in dronevehicle["comparisons"].items()
        },
        "dronevehicle_standard": (
            {
                key: {
                    "mean": value["equal_run_mean"],
                    "interval": value["conditional_run_by_block_interval"],
                }
                for key, value in dronevehicle_standard["comparisons"].items()
            }
            if dronevehicle_standard is not None
            else None
        ),
        "external_evaluators": {
            key: {
                "mean": value["equal_run_mean"],
                "interval": value["conditional_run_by_block_interval"],
            }
            for key, value in external_evaluators["comparisons"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
