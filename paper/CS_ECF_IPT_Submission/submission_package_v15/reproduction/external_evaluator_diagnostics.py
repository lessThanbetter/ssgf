"""Diagnose confidence-scale and selection behavior of external evaluators."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SSGF_DATA_ROOT", str(SCRIPT_ROOT)))
sys.path.insert(0, str(DATA_ROOT / "scripts"))

import analyze_csecf_vs_cdf as csecf  # noqa: E402
from local_ap_reference import load as local_load, pooled as local_pooled  # noqa: E402


MODELS = ("yolov9t_coco", "yolov10s_coco", "yolov9t_domain")
FILENAMES = {
    "vis": "vis",
    "cdf": "cdf",
    "pia": "piafusion",
    "spg": "spgfusion",
}


def _ap(stats: dict, indices: np.ndarray | None = None) -> float:
    if indices is None:
        indices = np.arange(stats["n"])
    return float(local_pooled(stats, np.asarray(indices, dtype=np.int64))[0])


def _top5(state: dict, image: int) -> float:
    confidence = np.sort(
        np.asarray(state["cf"][image], dtype=np.float64)
    )[::-1]
    return float(confidence[:5].mean()) if len(confidence) else -1.0


def _image_ap(state: dict, image: int) -> float:
    confidence = np.asarray(state["cf"][image], dtype=np.float64)
    flags = np.asarray(state["tp"][image], dtype=float)[:, 0]
    n_gt = len(state["gt"][image])
    if n_gt == 0 or len(confidence) == 0:
        return 0.0
    order = np.argsort(-confidence)
    flags = flags[order]
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


def _calibration(stats: dict) -> dict:
    tp = np.asarray(stats["tp_all"], dtype=float)[:, 0]
    confidence = np.asarray(stats["cf_all"], dtype=float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    rows = []
    for index in range(len(bins) - 1):
        if index == len(bins) - 2:
            mask = (confidence >= bins[index]) & (confidence <= bins[index + 1])
        else:
            mask = (confidence >= bins[index]) & (confidence < bins[index + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(tp[mask].mean())
        ece += count / len(confidence) * abs(accuracy - mean_confidence)
        rows.append({
            "lo": float(bins[index]),
            "hi": float(bins[index + 1]),
            "count": count,
            "mean_confidence": mean_confidence,
            "observed_positive_rate": accuracy,
        })
    return {
        "detection_count": int(len(confidence)),
        "detection_ece_10bin": float(ece),
        "detection_brier": float(np.mean((confidence - tp) ** 2)),
        "bins": rows,
    }


def _load_model(result_root: Path, model: str) -> dict:
    directory = (
        result_root / "domain_yolov9t_v3" / "per_image"
        if model == "yolov9t_domain"
        else result_root / "per_image"
    )
    return {
        candidate: local_load(directory / f"{model}__{filename}.npz")
        for candidate, filename in FILENAMES.items()
    }


def _two_sided_bootstrap_p(draws: np.ndarray) -> float:
    draws = np.asarray(draws, dtype=float)
    positive = int(np.sum(draws >= 0.0))
    negative = int(np.sum(draws <= 0.0))
    return min(
        1.0,
        2.0 * min(positive + 1, negative + 1) / (len(draws) + 1),
    )


def _holm(values: list[float]) -> list[float]:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for position, index in enumerate(order):
        running = max(
            running,
            min(1.0, values[index] * (len(values) - position)),
        )
        adjusted[index] = running
    return adjusted.tolist()


def _bh(values: list[float]) -> list[float]:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_position, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_position + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result_root = args.data_root / "results" / "independent_evaluators"

    output = {"models": {}, "multiple_comparisons": {}}
    for model in MODELS:
        states = _load_model(result_root, model)
        shielded, unshielded = csecf._decisions(states)
        shielded_stats = csecf._policy_stats(states, shielded)
        unshielded_stats = csecf._policy_stats(states, unshielded)
        visible_ap = _ap(states["vis"])
        cdf_ap = _ap(states["cdf"])
        shielded_ap = _ap(shielded_stats)
        fused_indices = [
            index for index, choice in enumerate(shielded) if choice != "vis"
        ]
        visible_top5 = np.asarray([
            _top5(states["vis"], image)
            for image in range(states["vis"]["n"])
        ])
        selected_top5 = np.asarray([
            _top5(states[shielded[image]], image)
            for image in range(states["vis"]["n"])
        ])
        ratios = np.asarray([
            _top5(states[shielded[image]], image)
            / max(_top5(states["vis"], image), 1e-12)
            for image in fused_indices
        ])
        visible_image_ap = np.asarray([
            _image_ap(states["vis"], image)
            for image in range(states["vis"]["n"])
        ])
        selected_image_ap = np.asarray([
            _image_ap(states[shielded[image]], image)
            for image in range(states["vis"]["n"])
        ])
        switched_delta = (
            selected_image_ap[fused_indices]
            - visible_image_ap[fused_indices]
        )

        groups = sorted(set(str(stem)[:3] for stem in states["vis"]["stems"]))
        per_scene = []
        for group in groups:
            indices = np.asarray([
                index
                for index, stem in enumerate(states["vis"]["stems"])
                if str(stem).startswith(group)
            ])
            delta = _ap(shielded_stats, indices) - _ap(states["vis"], indices)
            no_shield_delta = (
                _ap(unshielded_stats, indices) - _ap(states["vis"], indices)
            )
            per_scene.append({
                "scene": group,
                "n_images": int(len(indices)),
                "csecf_minus_visible": float(delta),
                "no_shield_minus_visible": float(no_shield_delta),
            })

        output["models"][model] = {
            "ap50": {
                "visible": visible_ap,
                "always_cdf": cdf_ap,
                "no_shield": _ap(unshielded_stats),
                "csecf": shielded_ap,
            },
            "selection": {
                "selected_counts": {
                    candidate: shielded.count(candidate)
                    for candidate in ("vis", "cdf", "pia", "spg")
                },
                "fused_fraction": len(fused_indices) / len(shielded),
                "mean_switch_ratio": float(ratios.mean()) if len(ratios) else None,
                "median_switch_ratio": float(np.median(ratios)) if len(ratios) else None,
            },
            "confidence": {
                "visible_top5_mean": float(visible_top5.mean()),
                "csecf_selected_top5_mean": float(selected_top5.mean()),
            },
            "calibration": {
                "visible": _calibration(states["vis"]),
                "csecf": _calibration(shielded_stats),
            },
            "switched_images": {
                "count": len(fused_indices),
                "mean_per_image_ap50_change": (
                    float(switched_delta.mean()) if len(switched_delta) else None
                ),
                "median_per_image_ap50_change": (
                    float(np.median(switched_delta)) if len(switched_delta) else None
                ),
                "fraction_improved": (
                    float(np.mean(switched_delta > 0))
                    if len(switched_delta)
                    else None
                ),
                "fraction_unchanged": (
                    float(np.mean(switched_delta == 0))
                    if len(switched_delta)
                    else None
                ),
                "fraction_degraded": (
                    float(np.mean(switched_delta < 0))
                    if len(switched_delta)
                    else None
                ),
            },
            "per_scene": per_scene,
            "positive_scenes": int(sum(
                row["csecf_minus_visible"] > 0 for row in per_scene
            )),
            "negative_scenes": int(sum(
                row["csecf_minus_visible"] < 0 for row in per_scene
            )),
        }

    p_values = {}
    for model_index, model in enumerate(MODELS):
        per_scene = output["models"][model]["per_scene"]
        gains = np.asarray([
            row["csecf_minus_visible"] for row in per_scene
        ])
        rng = np.random.default_rng(3300 + model_index)
        draws = np.asarray([
            rng.choice(gains, size=len(gains), replace=True).mean()
            for _ in range(100000)
        ])
        p_values[model] = _two_sided_bootstrap_p(draws)
    output["multiple_comparisons"] = {
        "family": "three external evaluator CS-ECF-minus-visible contrasts",
        "bootstrap_draws": 100000,
        "raw_p": p_values,
        "holm": dict(zip(
            MODELS, _holm([p_values[model] for model in MODELS])
        )),
        "bh": dict(zip(
            MODELS, _bh([p_values[model] for model in MODELS])
        )),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        model: {
            "fused_fraction": row["selection"]["fused_fraction"],
            "mean_switch_ratio": row["selection"]["mean_switch_ratio"],
            "visible_ece": row["calibration"]["visible"]["detection_ece_10bin"],
            "csecf_ece": row["calibration"]["csecf"]["detection_ece_10bin"],
            "switched_fraction_degraded": row["switched_images"]["fraction_degraded"],
            "positive_scenes": row["positive_scenes"],
            "negative_scenes": row["negative_scenes"],
        }
        for model, row in output["models"].items()
    }, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
