"""Report per-class effects of the frozen confidence-shield policy."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Code\_server_handoff_v2\ssgf\scripts")
from local_ap_reference import compute_ap  # noqa: E402


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULTS = ROOT / "results/selective_fusion"
HELPERS = runpy.run_path(
    r"D:\Code\_server_handoff_v2\tmp\confidence_candidate_selection.py"
)
rows_for = HELPERS["rows_for"]
score_confidence = HELPERS["confidence_score"]
PROFILES = {
    "llvip": [
        "yolo11",
        "agnostic",
        "yolo26_s0",
        "rtdetr",
        "yolo26_s1",
        "yolo26_s2",
        "yolo8_s0",
        "yolo8_s1",
    ],
    "m3fd": ["yolo26_m3fd6", "yolo8n_m3fd6"],
}
CLASS_NAMES = {
    "llvip": {0: "pedestrian"},
    "m3fd": {
        0: "people",
        1: "car",
        2: "bus",
        3: "motorcycle",
        4: "lamp",
        5: "truck",
    },
}


def choices_for(rows, n_images, margin):
    choices = []
    for image_index in range(n_images):
        values = {
            candidate: score_confidence(data, image_index, "top5")
            for candidate, data in rows.items()
        }
        selected = max(values, key=values.get)
        if selected != "vis" and values[selected] < margin * values["vis"]:
            selected = "vis"
        choices.append(selected)
    return choices


def class_ap50(rows, choices, class_ids, indices=None):
    tp, conf, pcls, tcls = [], [], [], []
    n_gt = rows["vis"]["n_gt"]
    image_indices = (
        range(len(choices)) if indices is None else np.asarray(indices)
    )
    for image_index in image_indices:
        image_index = int(image_index)
        candidate = choices[image_index]
        data = rows[candidate]
        mask = data["img_idx"] == image_index
        tp.append(data["tp"][mask])
        conf.append(data["conf"][mask])
        pcls.append(data["pcls"][mask])
        start = int(n_gt[:image_index].sum())
        stop = int(n_gt[:image_index + 1].sum())
        tcls.append(data["tcls"][start:stop])
    tp = np.concatenate(tp)
    conf = np.concatenate(conf)
    pcls = np.concatenate(pcls)
    tcls = np.concatenate(tcls)
    order = np.argsort(-conf)
    tp = tp[order]
    pcls = pcls[order]
    output = {}
    for class_id in class_ids:
        gt = tcls[tcls == class_id]
        selected = pcls == class_id
        if gt.size == 0 or selected.sum() == 0:
            output[str(class_id)] = 0.0
            continue
        fp = (1 - tp[selected, 0]).cumsum()
        positive = tp[selected, 0].cumsum()
        recall = positive / gt.size
        precision = positive / (positive + fp)
        output[str(class_id)] = round(
            compute_ap(recall, precision), 4
        )
    return output


def main():
    report = {}
    for profile, models in PROFILES.items():
        report[profile] = {}
        for model in models:
            rows = rows_for(profile, model)
            n_images = len(rows["vis"]["stems"])
            class_ids = sorted(CLASS_NAMES[profile])
            choices = {
                "visible": ["vis"] * n_images,
                "always_cdf": ["cdf"] * n_images,
                "top5_no_shield": choices_for(rows, n_images, 1.0),
                "confidence_shield": choices_for(rows, n_images, 1.5),
            }
            metrics = {
                policy: class_ap50(rows, policy_choices, class_ids)
                for policy, policy_choices in choices.items()
            }
            gains = {
                policy: {
                    class_id: round(
                        metrics[policy][class_id]
                        - metrics["visible"][class_id],
                        4,
                    )
                    for class_id in metrics[policy]
                }
                for policy in metrics
                if policy != "visible"
            }
            report[profile][model] = {
                "classes": {
                    str(class_id): CLASS_NAMES[profile][class_id]
                    for class_id in class_ids
                },
                "ap50": metrics,
                "gain_vs_visible": gains,
            }
            print(profile, model, report[profile][model], flush=True)
    output_path = RESULTS / "confidence_shield_class_diagnostics.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", output_path)


if __name__ == "__main__":
    main()
