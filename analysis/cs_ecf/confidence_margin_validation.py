"""Select confidence-shielding margin on validation data only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Code\_server_handoff_v2\ssgf\scripts")
from local_ap_reference import ap_per_class  # noqa: E402


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULTS = ROOT / "results/selective_fusion"
PROFILES = {
    "m3fd": {
        "models": ["yolo26_m3fd6", "yolo8n_m3fd6"],
        "cdf": RESULTS / "m3fd_tau_validation_6cls",
        "sota": RESULTS / "m3fd_sota_validation",
    },
    "llvip": {
        "models": ["yolo26n", "rtdetr"],
        "cdf": RESULTS / "validation_scores_pixelmean_cdf",
        "sota": RESULTS / "validation_scores_with_sota",
    },
}
CDF_FILE = "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_rows(profile, model):
    spec = PROFILES[profile]
    cdf_prefix = model
    sota_prefix = model
    return {
        "vis": load_npz(spec["cdf"] / f"{cdf_prefix}__vis.npz"),
        "cdf": load_npz(spec["cdf"] / f"{cdf_prefix}__{CDF_FILE}.npz"),
        "pia": load_npz(spec["sota"] / f"{sota_prefix}__piafusion.npz"),
        "spg": load_npz(spec["sota"] / f"{sota_prefix}__spgfusion.npz"),
    }


def score(data, image_index, policy):
    mask = data["img_idx"] == image_index
    confidence = np.sort(np.asarray(data["conf"][mask], np.float64))[::-1]
    if len(confidence) == 0:
        return -1.0
    if policy == "top5":
        return float(confidence[:5].mean())
    if policy == "sum_log":
        return float(np.log1p(confidence).sum())
    raise ValueError(policy)


def pooled(rows, choices):
    tp, conf, pcls, tcls = [], [], [], []
    n_gt = rows["vis"]["n_gt"]
    for image_index, candidate in enumerate(choices):
        data = rows[candidate]
        mask = data["img_idx"] == image_index
        tp.append(data["tp"][mask])
        conf.append(data["conf"][mask])
        pcls.append(data["pcls"][mask])
        start = int(n_gt[:image_index].sum())
        stop = int(n_gt[: image_index + 1].sum())
        tcls.append(data["tcls"][start:stop])
    return ap_per_class(
        np.concatenate(tp),
        np.concatenate(conf),
        np.concatenate(pcls),
        np.concatenate(tcls),
    )[0]


def main():
    policies = [
        ("top5", 1.05),
        ("top5", 1.10),
        ("top5", 1.20),
        ("top5", 1.30),
        ("top5", 1.50),
        ("sum_log", 1.05),
        ("sum_log", 1.10),
        ("sum_log", 1.20),
        ("sum_log", 1.30),
        ("sum_log", 1.50),
    ]
    report = {}
    for profile, spec in PROFILES.items():
        report[profile] = {}
        for model in spec["models"]:
            rows = load_rows(profile, model)
            n_images = len(rows["vis"]["stems"])
            baseline = pooled(rows, ["vis"] * n_images)
            model_report = {"visible_ap50": round(float(baseline), 4)}
            for base_policy, margin in policies:
                choices = []
                for image_index in range(n_images):
                    values = {
                        candidate: score(rows[candidate], image_index, base_policy)
                        for candidate in rows
                    }
                    selected = max(values, key=values.get)
                    if selected != "vis" and values[selected] < margin * values["vis"]:
                        selected = "vis"
                    choices.append(selected)
                selected_ap = pooled(rows, choices)
                name = f"{base_policy}_r{int(round(margin * 100)):03d}"
                model_report[name] = {
                    "gain": round(float(selected_ap - baseline), 4),
                    "counts": {
                        candidate: choices.count(candidate)
                        for candidate in rows
                    },
                }
                print(profile, model, name, model_report[name], flush=True)
            report[profile][model] = model_report
    output = RESULTS / "confidence_margin_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", output)


if __name__ == "__main__":
    main()
