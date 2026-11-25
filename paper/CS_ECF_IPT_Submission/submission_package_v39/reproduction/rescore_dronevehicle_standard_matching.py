"""Recompute DroneVehicle true-positive flags with confidence-ordered matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def xyxyxyxy2xywhr(points: np.ndarray) -> torch.Tensor:
    rotated = []
    for polygon in np.asarray(points, dtype=np.float32):
        (cx, cy), (width, height), angle_deg = cv2.minAreaRect(
            polygon.reshape(-1, 2)
        )
        theta = angle_deg / 180 * np.pi
        if width < height:
            width, height = height, width
            theta += np.pi / 2
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi
        rotated.append([cx, cy, width, height, theta])
    array = np.asarray(rotated, dtype=np.float32).reshape(-1, 5)
    return torch.from_numpy(array)


def _covariance(
    boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos, sin = c.cos(), c.sin()
    return (
        a * cos.pow(2) + b * sin.pow(2),
        a * sin.pow(2) + b * cos.pow(2),
        (a - b) * cos * sin,
    )


def batch_probiou(
    obb1: torch.Tensor,
    obb2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (
        value.squeeze(-1)[None]
        for value in obb2[..., :2].split(1, dim=-1)
    )
    a1, b1, c1 = _covariance(obb1)
    a2, b2, c2 = (
        value.squeeze(-1)[None] for value in _covariance(obb2)
    )
    t1 = (
        (
            (a1 + a2) * (y1 - y2).pow(2)
            + (b1 + b2) * (x1 - x2).pow(2)
        )
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.5
    t3 = (
        (
            (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)
        )
        / (
            4
            * (
                (a1 * b1 - c1.pow(2)).clamp_(0)
                * (a2 * b2 - c2.pow(2)).clamp_(0)
            ).sqrt()
            + eps
        )
        + eps
    ).log() * 0.5
    distance = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-distance).exp() + eps).sqrt()
    return 1 - hd


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
        xyxyxyxy2xywhr(pred_polygons),
        xyxyxyxy2xywhr(gt_polygons),
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
        default=ROOT / "results" / "prospective_20261029"
        / "dronevehicle" / "native_scores",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "prospective_20261029"
        / "dronevehicle" / "standard_matching",
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

    protocol = {
        "matching": "confidence-ordered one-to-one matching",
        "iou": "probabilistic rotated-box IoU",
        "iou_thresholds": IOU_THRESHOLDS.tolist(),
        "purpose": (
            "Sensitivity analysis for the all-at-once maximum-cardinality "
            "matching used in the native DroneVehicle score arrays."
        ),
        "files": outputs,
    }
    (args.out / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    print("WROTE", args.out, len(outputs), "score arrays")


if __name__ == "__main__":
    main()
