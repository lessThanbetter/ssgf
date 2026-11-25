"""Select one successful and one failed CS-ECF fused-input switch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CANDIDATES = {
    "CDF": "cdf",
    "PIAFusion": "piafusion",
    "SPGFusion": "spgfusion",
}


def image_ap50(tp: np.ndarray, confidence: np.ndarray, n_gt: int) -> float:
    if n_gt <= 0 or not len(confidence):
        return 0.0
    order = np.argsort(-confidence)
    tp_sorted = tp[order].astype(np.float64)
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(1.0 - tp_sorted)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    grid = np.linspace(0.0, 1.0, 101)
    sampled_precision = np.zeros_like(grid)
    for index, threshold in enumerate(grid):
        valid = recall >= threshold
        if valid.any():
            sampled_precision[index] = precision[valid].max()
    return float(sampled_precision.mean())


def top5_score(confidence: np.ndarray) -> float:
    if not len(confidence):
        return 0.0
    return float(np.sort(confidence)[-min(5, len(confidence)) :].mean())


def load_image_stats(path: Path) -> dict:
    state = np.load(path, allow_pickle=True)
    n_images = int(state["n_img"])
    stats = []
    for image_index in range(n_images):
        keep = state["img_idx"] == image_index
        confidence = state["conf"][keep]
        true_positives = state["tp"][keep, 0]
        stats.append(
            {
                "confidence": confidence,
                "score": top5_score(confidence),
                "ap50": image_ap50(
                    true_positives,
                    confidence,
                    int(state["n_gt"][image_index]),
                ),
                "true_positives": int(true_positives.sum()),
                "max_confidence": (
                    float(confidence.max()) if len(confidence) else -1.0
                ),
            }
        )
    return {"stems": [str(stem) for stem in state["stems"]], "images": stats}


def evaluate_sequence(root: Path, sequence: str, model: str) -> list[dict]:
    score_dir = root / f"llvip{sequence}" / "scores"
    paths = {
        "Visible": score_dir / f"{model}__vis.npz",
        **{
            name: score_dir / f"{model}__{suffix}.npz"
            for name, suffix in CANDIDATES.items()
        },
    }
    states = {
        name: load_image_stats(path)
        for name, path in paths.items()
    }
    stems = states["Visible"]["stems"]
    for name, state in states.items():
        if state["stems"] != stems:
            raise ValueError(f"{sequence}/{name}: stem order mismatch")

    rows = []
    for image_index, stem in enumerate(stems):
        fused = {
            name: state["images"][image_index]
            for name, state in states.items()
            if name != "Visible"
        }
        visible = states["Visible"]["images"][image_index]
        selected_name = max(fused, key=lambda name: fused[name]["score"])
        selected = fused[selected_name]
        switched = selected["score"] >= 1.5 * visible["score"]
        ap_gain = selected["ap50"] - visible["ap50"] if switched else 0.0
        rows.append(
            {
                "sequence": sequence,
                "stem": stem,
                "model": model,
                "selected": selected_name if switched else "Visible",
                "visible_score": visible["score"],
                "selected_score": selected["score"] if switched else visible["score"],
                "visible_ap50": visible["ap50"],
                "selected_ap50": selected["ap50"] if switched else visible["ap50"],
                "ap50_gain": ap_gain,
                "n_gt": int(np.load(paths["Visible"], allow_pickle=True)["n_gt"][image_index]),
                "visible_true_positives": visible["true_positives"],
                "selected_true_positives": (
                    selected["true_positives"] if switched
                    else visible["true_positives"]
                ),
                "visible_max_confidence": visible["max_confidence"],
                "selected_max_confidence": (
                    selected["max_confidence"] if switched
                    else visible["max_confidence"]
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path(
            r"D:\Code\_server_handoff_v2\ssgf\paper\CS_ECF_IPT_Submission"
            r"\submission_package_v24\evidence\prospective"
        ),
    )
    parser.add_argument("--model", default="yolo26_s0")
    parser.add_argument("--sequences", default="050,060,070")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            r"D:\Code\_server_handoff_v2\ssgf\paper\CS_ECF_IPT_Submission"
            r"\submission_package_v24\evidence\case_studies\selection.json"
        ),
    )
    args = parser.parse_args()

    rows = []
    for sequence in args.sequences.split(","):
        rows.extend(evaluate_sequence(args.evidence_root, sequence, args.model))
    switched = [
        row for row in rows
        if row["selected"] != "Visible" and row["n_gt"] >= 1
    ]
    visible_threshold = 0.25
    visible_switches = [
        row for row in switched
        if max(
            row["visible_max_confidence"],
            row["selected_max_confidence"],
        ) >= visible_threshold
    ]
    successes = [
        row for row in visible_switches
        if row["selected_true_positives"] > row["visible_true_positives"]
        and row["ap50_gain"] > 0
    ]
    failures = [
        row for row in visible_switches
        if row["selected_true_positives"] < row["visible_true_positives"]
        and row["ap50_gain"] < 0
    ]
    if not successes or not failures:
        raise SystemExit(
            "No confidence-visible success and failure pair with changed "
            "matched true-positive counts"
        )

    success = max(
        successes,
        key=lambda row: (
            row["ap50_gain"],
            row["selected_true_positives"]
            - row["visible_true_positives"],
        ),
    )
    failure = min(
        failures,
        key=lambda row: (
            row["ap50_gain"],
            row["selected_true_positives"]
            - row["visible_true_positives"],
        ),
    )
    payload = {
        "model": args.model,
        "selection_rule": "Top-five confidence with a visible-relative margin of 1.5.",
        "selection_method": (
            "Among switched images with a detector response at or above the "
            "display threshold in either input, select the largest positive "
            "AP50 change for which the fused input adds a matched true "
            "positive, and the largest negative change for which it loses a "
            "matched true positive."
        ),
        "display_confidence_threshold": visible_threshold,
        "success": success,
        "failure": failure,
        "counts": {
            "images": len(rows),
            "switched": sum(row["selected"] != "Visible" for row in rows),
            "improved": sum(
                row["selected"] != "Visible" and row["ap50_gain"] > 0 for row in rows
            ),
            "degraded": sum(
                row["selected"] != "Visible" and row["ap50_gain"] < 0 for row in rows
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
