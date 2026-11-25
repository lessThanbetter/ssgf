"""Analyze CS-ECF using native DroneVehicle OBB score arrays."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_ap_reference import load, pooled  # noqa: E402


TOP_K = 5
MARGIN = 1.5
BOOTSTRAP_DRAWS = 10000
SEED = 2917


def topk(values: np.ndarray) -> float:
    if len(values) == 0:
        return -1.0
    return float(np.sort(values)[-TOP_K:].mean())


def decisions(states: dict[str, dict], margin: float = MARGIN) -> tuple[list[str], list[str]]:
    selected = []
    unshielded = []
    for image in range(states["vis"]["n"]):
        scores = {
            name: topk(state["cf"][image])
            for name, state in states.items()
        }
        best = max(("vis", "cdf", "pia", "spg"), key=lambda name: scores[name])
        unshielded.append(best)
        if best != "vis" and scores[best] >= margin * scores["vis"]:
            selected.append(best)
        else:
            selected.append("vis")
    return selected, unshielded


def policy_stats(states: dict[str, dict], selected: list[str]) -> dict:
    reference = states["vis"]
    tp = []
    conf = []
    pred_cls = []
    gt = []
    det_counts = []
    gt_counts = []
    for image, candidate in enumerate(selected):
        state = states[candidate]
        tp.append(state["tp"][image])
        conf.append(state["cf"][image])
        pred_cls.append(state["cl"][image])
        gt.append(state["gt"][image])
        det_counts.append(len(state["cf"][image]))
        gt_counts.append(len(state["gt"][image]))
    det_counts_array = np.asarray(det_counts, dtype=np.int64)
    gt_counts_array = np.asarray(gt_counts, dtype=np.int64)
    return {
        "n": reference["n"],
        "stems": reference["stems"],
        "tp_all": np.concatenate(tp),
        "cf_all": np.concatenate(conf),
        "cl_all": np.concatenate(pred_cls),
        "gt_all": np.concatenate(gt),
        "det_counts": det_counts_array,
        "det_offsets": np.concatenate(([0], np.cumsum(det_counts_array)[:-1])),
        "gt_counts": gt_counts_array,
        "gt_offsets": np.concatenate(([0], np.cumsum(gt_counts_array)[:-1])),
    }


def groups(stems: list[str]) -> list[np.ndarray]:
    labels = []
    for stem in stems:
        match = re.search(r"test_(\d+)", stem)
        if match is None:
            raise ValueError(stem)
        labels.append(str((int(match.group(1)) - 1) // 100))
    labels = np.asarray(labels)
    return [
        np.flatnonzero(labels == label)
        for label in sorted(set(labels), key=int)
    ]


def ap50(stats: dict, indices: np.ndarray) -> float:
    return float(pooled(stats, indices)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    rng = np.random.default_rng(SEED)
    for model in args.models:
        states = {
            candidate: load(args.scores / f"{model}__{candidate}.npz")
            for candidate in ("vis", "cdf", "pia", "spg")
        }
        selected, unshielded = decisions(states)
        selections = {
            "visible": states["vis"],
            "always_cdf": states["cdf"],
            "no_shield": policy_stats(states, unshielded),
            "cs_ecf": policy_stats(states, selected),
        }
        blocks = groups(states["vis"]["stems"])
        row = {"per_block": {}, "switch_fraction": float(
            np.mean([choice != "vis" for choice in selected])
        )}
        for policy, stats in selections.items():
            values = np.asarray([
                ap50(stats, block) - ap50(states["vis"], block)
                for block in blocks
            ])
            row["per_block"][policy] = {
                "gains": values.tolist(),
                "mean": float(values.mean()),
                "worst": float(values.min()),
                "positive_blocks": int(np.sum(values > 0)),
            }
        rows[model] = row

    for model, row in rows.items():
        results = {}
        for policy in ("always_cdf", "no_shield", "cs_ecf"):
            gains = np.asarray(row["per_block"][policy]["gains"])
            draws = [
                float(gains[
                    rng.integers(0, len(gains), size=len(gains))
                ].mean())
                for _ in range(BOOTSTRAP_DRAWS)
            ]
            results[policy] = {
                "mean_gain": float(gains.mean()),
                "worst_block": float(gains.min()),
                "positive_blocks": int(np.sum(gains > 0)),
                "conditional_block_interval": [
                    float(value)
                    for value in np.percentile(draws, [2.5, 97.5])
                ],
            }
        row["summary"] = results

    aggregate = {}
    for policy in ("always_cdf", "no_shield", "cs_ecf"):
        model_means = {
            model: row["per_block"][policy]["mean"]
            for model, row in rows.items()
        }
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            sampled_models = rng.choice(
                list(rows),
                size=len(rows),
                replace=True,
            )
            sampled_values = []
            for model in sampled_models:
                gains = np.asarray(rows[model]["per_block"][policy]["gains"])
                sampled_blocks = rng.integers(
                    0,
                    len(gains),
                    size=len(gains),
                )
                sampled_values.append(float(gains[sampled_blocks].mean()))
            draws.append(float(np.mean(sampled_values)))
        aggregate[policy] = {
            "equal_run_mean": float(np.mean(list(model_means.values()))),
            "per_run": model_means,
            "worst_run": float(min(model_means.values())),
            "positive_runs": int(sum(value > 0 for value in model_means.values())),
            "conditional_run_by_block_interval": [
                float(value)
                for value in np.percentile(draws, [2.5, 97.5])
            ],
        }

    rows["aggregate"] = aggregate
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
