"""Select visible/fused inputs per image using label-free detector confidence."""

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
        "candidates": {
            "vis": (RESULTS / "m3fd_6cls_multiclass", "yolo26_m3fd6"),
            "cdf": (
                RESULTS / "m3fd_6cls_multiclass",
                "yolo26_m3fd6",
            ),
            "pia": (RESULTS / "m3fd_sota_scores", "yolo26_m3fd6"),
            "spg": (RESULTS / "m3fd_sota_scores", "yolo26_m3fd6"),
        },
        "files": {
            "vis": "vis",
            "cdf": "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05",
            "pia": "piafusion",
            "spg": "spgfusion",
        },
    },
    "dronevehicle": {
        "models": ["yolo11x_obb"],
        "candidates": {},
        "files": {
            "vis": "vis",
            "cdf": "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05",
            "pia": "piafusion",
            "spg": "spgfusion",
        },
    },
    "llvip": {
        "models": [
            "yolo11",
            "agnostic",
            "yolo26_s0",
            "rtdetr",
            "yolo26_s1",
            "yolo26_s2",
            "yolo8_s0",
            "yolo8_s1",
        ],
        "candidates": {},
        "files": {
            "vis": "vis",
            "cdf": "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05",
            "pia": "piafusion",
            "spg": "spgfusion",
        },
    },
}


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def rows_for(profile, model):
    if profile == "llvip":
        cdf_directory = RESULTS / "test_scores_pixelmean_cdf"
        cdf_prefix = {
            "yolo26_s0": "yolo26",
            "yolo8_s0": "yolo8n_s0",
            "yolo8_s1": "yolo8n_s1",
        }.get(model, model)
        sota_specs = {
            "yolo11": (ROOT / "_deposit/ssgf-llvip-fusion-detection-v1/data/per_image", "vis_split"),
            "agnostic": (ROOT / "_deposit/ssgf-llvip-fusion-detection-v1/data/per_image", "agn_split"),
            "yolo26_s0": (ROOT / "results/clean_r2/full_16/per_image", "yolo26n_clean"),
            "rtdetr": (ROOT / "results/clean_r2/full_16/per_image", "rtdetr_l_clean"),
            "yolo26_s1": (RESULTS / "test_scores_sota_missing", "yolo26_s1"),
            "yolo26_s2": (RESULTS / "test_scores_sota_missing", "yolo26_s2"),
            "yolo8_s0": (RESULTS / "test_scores_sota_missing", "yolo8n_s0"),
            "yolo8_s1": (RESULTS / "test_scores_sota_missing", "yolo8n_s1"),
        }
        sota_directory, sota_prefix = sota_specs[model]
        files = PROFILES[profile]["files"]
        return {
            "vis": load_npz(cdf_directory / f"{cdf_prefix}__vis.npz"),
            "cdf": load_npz(cdf_directory / f"{cdf_prefix}__{files['cdf']}.npz"),
            "pia": load_npz(sota_directory / f"{sota_prefix}__piafusion.npz"),
            "spg": load_npz(sota_directory / f"{sota_prefix}__spgfusion.npz"),
        }
    if profile == "m3fd":
        directories = {
            "vis": RESULTS / "m3fd_6cls_multiclass",
            "cdf": RESULTS / "m3fd_6cls_multiclass",
            "pia": RESULTS / "m3fd_sota_scores",
            "spg": RESULTS / "m3fd_sota_scores",
        }
        prefix = model
    else:
        directories = {
            "vis": RESULTS / "dronevehicle_scores",
            "cdf": RESULTS / "dronevehicle_scores",
            "pia": RESULTS / "dronevehicle_sota_scores",
            "spg": RESULTS / "dronevehicle_sota_scores",
        }
        prefix = model
    files = PROFILES[profile]["files"]
    return {
        candidate: load_npz(directory / f"{prefix}__{files[candidate]}.npz")
        for candidate, directory in directories.items()
    }


def pooled(rows, choices):
    tp, conf, pcls, tcls = [], [], [], []
    for image_index, candidate in enumerate(choices):
        data = rows[candidate]
        mask = data["img_idx"] == image_index
        tp.append(data["tp"][mask])
        conf.append(data["conf"][mask])
        pcls.append(data["pcls"][mask])
        tcls.append(data["tcls"][data["n_gt"][: image_index + 1].sum() - data["n_gt"][image_index] : data["n_gt"][: image_index + 1].sum()])
    return ap_per_class(
        np.concatenate(tp),
        np.concatenate(conf),
        np.concatenate(pcls),
        np.concatenate(tcls),
    )[0]


def confidence_score(data, image_index, policy):
    mask = data["img_idx"] == image_index
    confidence = np.sort(np.asarray(data["conf"][mask], np.float64))[::-1]
    if len(confidence) == 0:
        return -1.0
    if policy == "top1":
        return float(confidence[0])
    if policy == "top5":
        return float(confidence[:5].mean())
    if policy == "mean":
        return float(confidence.mean())
    if policy == "count01":
        return float(np.sum(confidence >= 0.01))
    if policy == "count05":
        return float(np.sum(confidence >= 0.05))
    if policy == "sum_log":
        return float(np.log1p(confidence).sum())
    raise ValueError(policy)


def main():
    policies = (
        "top1",
        "top5",
        "mean",
        "count01",
        "count05",
        "sum_log",
        "top5_r105",
        "top5_r110",
        "top5_r120",
        "top5_r150",
        "sumlog_r105",
        "sumlog_r110",
        "sumlog_r120",
        "sumlog_r150",
    )
    report = {}
    for profile, spec in PROFILES.items():
        report[profile] = {}
        for model in spec["models"]:
            rows = rows_for(profile, model)
            stems = [str(stem) for stem in rows["vis"]["stems"]]
            baseline = pooled(rows, ["vis"] * len(stems))
            model_report = {"visible_ap50": round(float(baseline), 4)}
            for policy in policies:
                choices = []
                for image_index in range(len(stems)):
                    base_policy = policy
                    margin = 1.0
                    if "_r" in policy:
                        base_policy, ratio_text = policy.rsplit("_r", 1)
                        margin = int(ratio_text) / 100.0
                    base_policy = base_policy.replace("sumlog", "sum_log")
                    scores = {
                        candidate: confidence_score(
                            rows[candidate], image_index, base_policy
                        )
                        for candidate in rows
                    }
                    selected = max(scores, key=scores.get)
                    if (
                        selected != "vis"
                        and scores[selected] < margin * scores["vis"]
                    ):
                        selected = "vis"
                    choices.append(selected)
                selected = pooled(rows, choices)
                model_report[policy] = {
                    "ap50": round(float(selected), 4),
                    "gain": round(float(selected - baseline), 4),
                    "counts": {
                        candidate: choices.count(candidate)
                        for candidate in rows
                    },
                }
            report[profile][model] = model_report
            print(profile, model, model_report, flush=True)
    output = RESULTS / "confidence_candidate_selection.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("WROTE", output)


if __name__ == "__main__":
    main()
