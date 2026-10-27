"""Combine both DroneVehicle test checkpoints with paired group bootstrap."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Code\_server_handoff_v2\ssgf\scripts")
from local_ap_reference import ap_per_class  # noqa: E402


HELPERS = runpy.run_path(
    r"D:\Code\_server_handoff_v2\tmp\analyze_dronevehicle_test_csf.py"
)
load_npz = HELPERS["load_npz"]
choices_for = HELPERS["choices_for"]
FILES = HELPERS["FILES"]
CANDIDATES = HELPERS["CANDIDATES"]
ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULT_ROOT = ROOT / "results/selective_fusion"
MODELS = {
    "best": (RESULT_ROOT / "dronevehicle_test_scores", "yolo11x_obb"),
    "last": (RESULT_ROOT / "dronevehicle_test_scores_last", "yolo11x_obb_last"),
}
DRAWS = 500


def pooled(rows, choices, indices):
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


def load_model(directory, prefix):
    return {
        candidate: load_npz(
            directory / f"{prefix}__{FILES[candidate]}.npz"
        )
        for candidate in CANDIDATES
    }


def main():
    rng = np.random.default_rng(4077)
    models = {
        name: load_model(directory, prefix)
        for name, (directory, prefix) in MODELS.items()
    }
    first_rows = models["best"]["vis"]
    n_images = len(first_rows["stems"])
    stems = np.asarray([str(value) for value in first_rows["stems"]])
    groups = np.asarray(
        [(int(stem.split("_")[-1]) - 1) // 100 for stem in stems]
    )
    group_indices = {
        int(group): np.flatnonzero(groups == group)
        for group in sorted(set(groups))
    }
    all_indices = np.arange(n_images)

    choices = {
        name: choices_for(rows, 1.5) for name, rows in models.items()
    }
    no_shield_choices = {
        name: choices_for(rows, 1.0) for name, rows in models.items()
    }
    point = {}
    for name, rows in models.items():
        visible = pooled(rows, ["vis"] * n_images, all_indices)
        shielded = pooled(rows, choices[name], all_indices)
        no_shield = pooled(rows, no_shield_choices[name], all_indices)
        per_group = {}
        for group, indices in group_indices.items():
            group_visible = pooled(rows, ["vis"] * n_images, indices)
            group_shielded = pooled(rows, choices[name], indices)
            per_group[str(group)] = {
                "n": len(indices),
                "visible_ap50": round(group_visible, 4),
                "shielded_ap50": round(group_shielded, 4),
                "gain": round(group_shielded - group_visible, 4),
            }
        point[name] = {
            "visible_ap50": round(visible, 4),
            "shielded_ap50": round(shielded, 4),
            "shielded_gain": round(shielded - visible, 4),
            "no_shield_gain": round(no_shield - visible, 4),
            "shielded_selected_counts": {
                candidate: choices[name].count(candidate)
                for candidate in CANDIDATES
            },
            "per_group": per_group,
        }

    image_draws = {name: [] for name in models}
    no_shield_image_draws = {name: [] for name in models}
    group_draws = {name: [] for name in models}
    no_shield_group_draws = {name: [] for name in models}
    for _ in range(DRAWS):
        indices = rng.choice(all_indices, n_images, replace=True)
        picked_groups = rng.choice(
            sorted(group_indices), len(group_indices), replace=True
        )
        group_sample = np.concatenate(
            [group_indices[group] for group in picked_groups]
        )
        for name, rows in models.items():
            visible_image = pooled(rows, ["vis"] * n_images, indices)
            image_draws[name].append(
                pooled(rows, choices[name], indices) - visible_image
            )
            no_shield_image_draws[name].append(
                pooled(rows, no_shield_choices[name], indices)
                - visible_image
            )
            visible_group = pooled(rows, ["vis"] * n_images, group_sample)
            group_draws[name].append(
                pooled(rows, choices[name], group_sample) - visible_group
            )
            no_shield_group_draws[name].append(
                pooled(rows, no_shield_choices[name], group_sample)
                - visible_group
            )

    image_matrix = np.asarray(list(image_draws.values()))
    no_shield_image_matrix = np.asarray(
        list(no_shield_image_draws.values())
    )
    group_matrix = np.asarray(list(group_draws.values()))
    no_shield_group_matrix = np.asarray(list(no_shield_group_draws.values()))
    combined_image = image_matrix.mean(axis=0)
    combined_group = group_matrix.mean(axis=0)
    combined_no_shield_image = no_shield_image_matrix.mean(axis=0)
    combined_no_shield_group = no_shield_group_matrix.mean(axis=0)

    checkpoint_gains = np.asarray(
        [point[name]["shielded_gain"] for name in models]
    )
    no_shield_checkpoint_gains = np.asarray(
        [point[name]["no_shield_gain"] for name in models]
    )
    report = {
        "dataset": "DroneVehicle official test",
        "independent_status": "confirmatory test; three groups fixed before evaluation",
        "n_images": n_images,
        "groups": sorted(group_indices),
        "checkpoint_results": point,
        "combined": {
            "mean_shielded_gain": round(
                float(checkpoint_gains.mean()), 4
            ),
            "min_checkpoint_gain": round(
                float(checkpoint_gains.min()), 4
            ),
            "max_checkpoint_gain": round(
                float(checkpoint_gains.max()), 4
            ),
            "mean_no_shield_gain": round(
                float(no_shield_checkpoint_gains.mean()), 4
            ),
            "image_ci95": [
                round(float(value), 4)
                for value in np.percentile(combined_image, [2.5, 97.5])
            ],
            "group_ci95": [
                round(float(value), 4)
                for value in np.percentile(combined_group, [2.5, 97.5])
            ],
            "no_shield_image_ci95": [
                round(float(value), 4)
                for value in np.percentile(
                    combined_no_shield_image, [2.5, 97.5]
                )
            ],
            "no_shield_group_ci95": [
                round(float(value), 4)
                for value in np.percentile(
                    combined_no_shield_group, [2.5, 97.5]
                )
            ],
        },
    }
    output = RESULT_ROOT / "dronevehicle_test_combined.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("WROTE", output)


if __name__ == "__main__":
    main()
