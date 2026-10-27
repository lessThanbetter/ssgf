"""Bootstrap CDF on the DroneVehicle official-validation subset."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
SCORES = ROOT / "results/selective_fusion/dronevehicle_scores"
CANDIDATE = "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"
DRAWS = 1000
HELPERS = runpy.run_path(
    str(Path(r"D:\Code\_server_handoff_v2\tmp\selective_fusion_cv_v3.py"))
)
load_npz = HELPERS["load_npz"]
sys.path.insert(0, r"D:\Code\_server_handoff_v2\ssgf\scripts")
from local_ap_reference import ap_per_class  # noqa: E402


def pooled(data, indices):
    tp, conf, pcls, tcls = [], [], [], []
    n_gt = data["n_gt"]
    for image_index in indices:
        image_index = int(image_index)
        mask = data["img_idx"] == image_index
        tp.append(data["tp"][mask])
        conf.append(data["conf"][mask])
        pcls.append(data["pcls"][mask])
        start = int(n_gt[:image_index].sum())
        stop = int(n_gt[:image_index + 1].sum())
        tcls.append(data["tcls"][start:stop])
    return float(
        ap_per_class(
            np.concatenate(tp),
            np.concatenate(conf),
            np.concatenate(pcls),
            np.concatenate(tcls),
        )[0]
    )


def main():
    visible = load_npz(SCORES / "yolo11x_obb__vis.npz")
    fused = load_npz(SCORES / f"yolo11x_obb__{CANDIDATE}.npz")
    stems = np.asarray([str(stem) for stem in visible["stems"]])
    groups = np.asarray(
        [(int(stem.removeprefix("val_")) - 1) // 100 for stem in stems]
    )
    group_indices = {
        group: np.flatnonzero(groups == group)
        for group in sorted(set(groups))
    }
    all_indices = np.arange(len(stems))
    baseline = pooled(visible, all_indices)
    selected = pooled(fused, all_indices)
    rng = np.random.default_rng(2213)
    image_draws, group_draws = [], []
    for _ in range(DRAWS):
        image_indices = rng.choice(all_indices, len(all_indices), replace=True)
        image_draws.append(
            pooled(fused, image_indices) - pooled(visible, image_indices)
        )
        picked_groups = rng.choice(
            sorted(group_indices), len(group_indices), replace=True
        )
        group_sample = np.concatenate(
            [group_indices[group] for group in picked_groups]
        )
        group_draws.append(
            pooled(fused, group_sample) - pooled(visible, group_sample)
        )
    per_group = {}
    for group, indices in group_indices.items():
        group_baseline = pooled(visible, indices)
        group_selected = pooled(fused, indices)
        per_group[str(group)] = {
            "n": len(indices),
            "visible_ap50": round(group_baseline, 4),
            "cdf_ap50": round(group_selected, 4),
            "gain": round(group_selected - group_baseline, 4),
        }
    report = {
        "candidate": CANDIDATE,
        "n_images": len(stems),
        "visible_ap50": round(baseline, 4),
        "cdf_ap50": round(selected, 4),
        "gain": round(selected - baseline, 4),
        "image_ci95": [
            round(float(value), 4)
            for value in np.percentile(image_draws, [2.5, 97.5])
        ],
        "group_ci95": [
            round(float(value), 4)
            for value in np.percentile(group_draws, [2.5, 97.5])
        ],
        "positive_groups": sum(
            value["gain"] > 0 for value in per_group.values()
        ),
        "groups": per_group,
    }
    output = ROOT / "results/selective_fusion/dronevehicle_bootstrap.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
