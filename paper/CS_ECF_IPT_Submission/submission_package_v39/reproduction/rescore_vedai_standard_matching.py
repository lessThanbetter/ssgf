"""Recompute VEDAI true-positive flags with confidence-ordered matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def confidence_ordered_match(
    pred_polygons: np.ndarray,
    pred_classes: np.ndarray,
    prediction_confidence: np.ndarray,
    gt_polygons: np.ndarray,
    gt_classes: np.ndarray,
) -> np.ndarray:
    correct = np.zeros(
        (len(pred_polygons), len(IOU_THRESHOLDS)),
        dtype=bool,
    )
    if len(pred_polygons) == 0 or len(gt_polygons) == 0:
        return correct
    overlaps = batch_probiou(
        xyxyxyxy2xywhr(torch.from_numpy(pred_polygons)),
        xyxyxyxy2xywhr(torch.from_numpy(gt_polygons)),
    ).numpy()
    same_class = pred_classes[:, None] == gt_classes[None, :]
    order = np.argsort(-prediction_confidence, kind="stable")
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
        unmatched = set(range(len(gt_polygons)))
        for prediction_index in order:
            eligible = [
                gt_index
                for gt_index in unmatched
                if same_class[prediction_index, gt_index]
                and overlaps[prediction_index, gt_index] >= threshold
            ]
            if not eligible:
                continue
            gt_index = max(
                eligible,
                key=lambda index: overlaps[prediction_index, index],
            )
            unmatched.remove(gt_index)
            correct[prediction_index, threshold_index] = True
    return correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "evidence"
        / "vedai"
        / "test_scores",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "evidence"
        / "vedai"
        / "standard_matching",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    outputs = []
    for source in sorted(args.source.glob("*.npz")):
        state = np.load(source, allow_pickle=True)
        n_images = int(state["n_img"])
        tp_rows = []
        for image_index in range(n_images):
            prediction_mask = state["poly_img_idx"] == image_index
            ground_truth_mask = state["gt_img_idx"] == image_index
            tp_rows.append(
                confidence_ordered_match(
                    state["polygons"][prediction_mask],
                    state["pcls"][prediction_mask],
                    state["conf"][prediction_mask],
                    state["gt_polygons"][ground_truth_mask],
                    state["tcls"][ground_truth_mask],
                )
            )
        output = args.out / source.name
        np.savez_compressed(
            output,
            tp=np.concatenate(tp_rows, axis=0),
            conf=state["conf"],
            pcls=state["pcls"],
            img_idx=state["img_idx"],
            tcls=state["tcls"],
            polygons=state["polygons"],
            poly_img_idx=state["poly_img_idx"],
            gt_polygons=state["gt_polygons"],
            gt_img_idx=state["gt_img_idx"],
            stems=state["stems"],
            n_img=state["n_img"],
            n_gt=state["n_gt"],
        )
        outputs.append(output.name)

    (args.out / "protocol.json").write_text(
        json.dumps(
            {
                "matching": "confidence-ordered one-to-one matching",
                "iou": "probabilistic rotated-box IoU",
                "iou_thresholds": IOU_THRESHOLDS.tolist(),
                "purpose": (
                    "Sensitivity analysis for the all-at-once "
                    "maximum-cardinality matching used in the primary scores."
                ),
                "files": outputs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("WROTE", args.out, len(outputs), "score arrays")


if __name__ == "__main__":
    main()
