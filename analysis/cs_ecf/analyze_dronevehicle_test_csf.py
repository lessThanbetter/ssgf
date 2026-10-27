"""Evaluate the frozen CS-ECF policy on the independent DroneVehicle test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Code\_server_handoff_v2\ssgf\scripts")
from local_ap_reference import ap_per_class  # noqa: E402


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
SCORES = ROOT / "results/selective_fusion/dronevehicle_test_scores"
PREFIX = "yolo11x_obb"
CANDIDATES = ("vis", "cdf", "pia", "spg")
FILES = {
    "vis": "vis",
    "cdf": "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05",
    "pia": "piafusion",
    "spg": "spgfusion",
}
DRAWS = 500


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def pooled(rows, choices, indices=None):
    if indices is None:
        indices = range(len(choices))
    tp, conf, pcls, tcls = [], [], [], []
    n_gt = rows["vis"]["n_gt"]
    for image_index in indices:
        image_index = int(image_index)
        data = rows[choices[image_index]]
        mask = data["img_idx"] == image_index
        tp.append(data["tp"][mask])
        conf.append(data["conf"][mask])
        pcls.append(data["pcls"][mask])
        start = int(n_gt[:image_index].sum())
        stop = int(n_gt[: image_index + 1].sum())
        tcls.append(data["tcls"][start:stop])
    return float(
        ap_per_class(
            np.concatenate(tp),
            np.concatenate(conf),
            np.concatenate(pcls),
            np.concatenate(tcls),
        )[0]
    )


def confidence(data, image_index):
    mask = data["img_idx"] == image_index
    values = np.sort(np.asarray(data["conf"][mask], np.float64))[::-1]
    if len(values) == 0:
        return -1.0
    return float(values[:5].mean())


def choices_for(rows, margin):
    choices = []
    n_images = len(rows["vis"]["stems"])
    for image_index in range(n_images):
        scores = {
            candidate: confidence(rows[candidate], image_index)
            for candidate in CANDIDATES
        }
        selected = max(scores, key=scores.get)
        if (
            selected != "vis"
            and scores[selected] < margin * scores["vis"]
        ):
            selected = "vis"
        choices.append(selected)
    return choices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--label", default="DroneVehicle official test")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/selective_fusion/dronevehicle_test_summary.json"
        ),
    )
    args = parser.parse_args()
    rows = {
        candidate: load_npz(
            args.scores / f"{args.prefix}__{FILES[candidate]}.npz"
        )
        for candidate in CANDIDATES
    }
    n_images = len(rows["vis"]["stems"])
    stems = np.asarray([str(value) for value in rows["vis"]["stems"]])
    groups = np.asarray(
        [(int(stem.split("_")[-1]) - 1) // 100 for stem in stems]
    )
    group_indices = {
        int(group): np.flatnonzero(groups == group)
        for group in sorted(set(groups))
    }
    all_indices = np.arange(n_images)
    visible = pooled(rows, ["vis"] * n_images)
    fixed = {
        candidate: pooled(rows, [candidate] * n_images)
        for candidate in CANDIDATES
    }
    shield_choices = choices_for(rows, 1.5)
    no_shield_choices = choices_for(rows, 1.0)
    shield = pooled(rows, shield_choices)
    no_shield = pooled(rows, no_shield_choices)

    rng = np.random.default_rng(7761)
    image_gains, no_shield_image_gains = [], []
    group_gains, no_shield_group_gains = [], []
    for _ in range(DRAWS):
        indices = rng.choice(all_indices, n_images, replace=True)
        visible_image = pooled(rows, ["vis"] * n_images, indices)
        image_gains.append(
            pooled(rows, shield_choices, indices) - visible_image
        )
        no_shield_image_gains.append(
            pooled(rows, no_shield_choices, indices) - visible_image
        )
        picked_groups = rng.choice(
            sorted(group_indices), len(group_indices), replace=True
        )
        group_sample = np.concatenate(
            [group_indices[group] for group in picked_groups]
        )
        visible_group = pooled(
            rows, ["vis"] * n_images, group_sample
        )
        group_gains.append(
            pooled(rows, shield_choices, group_sample) - visible_group
        )
        no_shield_group_gains.append(
            pooled(rows, no_shield_choices, group_sample) - visible_group
        )

    report = {
        "dataset": args.label,
        "selection": "three consecutive groups fixed before evaluation",
        "n_images": n_images,
        "groups": sorted(group_indices),
        "policy": {
            "top_k": 5,
            "visible_margin": 1.5,
            "candidates": list(CANDIDATES),
        },
        "visible_ap50": round(visible, 4),
        "fixed_ap50": {
            candidate: round(value, 4)
            for candidate, value in fixed.items()
        },
        "fixed_gain": {
            candidate: round(value - visible, 4)
            for candidate, value in fixed.items()
        },
        "no_shield_ap50": round(no_shield, 4),
        "no_shield_gain": round(no_shield - visible, 4),
        "shielded_ap50": round(shield, 4),
        "shielded_gain": round(shield - visible, 4),
        "shielded_image_ci95": [
            round(float(value), 4)
            for value in np.percentile(image_gains, [2.5, 97.5])
        ],
        "shielded_group_ci95": [
            round(float(value), 4)
            for value in np.percentile(group_gains, [2.5, 97.5])
        ],
        "no_shield_image_ci95": [
            round(float(value), 4)
            for value in np.percentile(no_shield_image_gains, [2.5, 97.5])
        ],
        "no_shield_group_ci95": [
            round(float(value), 4)
            for value in np.percentile(no_shield_group_gains, [2.5, 97.5])
        ],
        "shielded_selected_counts": {
            candidate: shield_choices.count(candidate)
            for candidate in CANDIDATES
        },
        "no_shield_selected_counts": {
            candidate: no_shield_choices.count(candidate)
            for candidate in CANDIDATES
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
