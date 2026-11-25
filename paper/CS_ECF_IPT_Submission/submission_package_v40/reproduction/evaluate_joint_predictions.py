"""Re-score COCO-format joint-detector predictions with the study AP protocol."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from local_ap_reference import ap_per_class


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    x1 = np.maximum(left[:, None, 0], right[None, :, 0])
    y1 = np.maximum(left[:, None, 1], right[None, :, 1])
    x2 = np.minimum(left[:, None, 2], right[None, :, 2])
    y2 = np.minimum(left[:, None, 3], right[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    left_area = np.clip(left[:, 2] - left[:, 0], 0, None) * np.clip(
        left[:, 3] - left[:, 1], 0, None
    )
    right_area = np.clip(right[:, 2] - right[:, 0], 0, None) * np.clip(
        right[:, 3] - right[:, 1], 0, None
    )
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def match(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> np.ndarray:
    correct = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return correct
    iou = pairwise_iou(pred_boxes, gt_boxes)
    same_class = pred_classes[:, None] == gt_classes[None, :]
    for index, threshold in enumerate(IOU_THRESHOLDS):
        valid = np.logical_and(iou >= threshold, same_class)
        if valid.any():
            rows, columns = linear_sum_assignment(np.logical_not(valid))
            correct[rows, index] = valid[rows, columns]
    return correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-manifest", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--image-dir", type=Path)
    args = parser.parse_args()

    cookies = json.loads(args.annotations.read_text(encoding="utf-8"))
    prediction_id_by_coco_id: dict[int, int] = {}
    for image in cookies["images"]:
        coco_id = int(image["id"])
        file_name = image.get("file_name_RGB") or image.get("file_name") or ""
        stem = Path(file_name).stem
        prediction_id_by_coco_id[coco_id] = int(stem) if stem.isdigit() else coco_id

    ground_truth: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    image_sizes: dict[int, tuple[int, int]] = {}
    if args.labels:
        if args.image_dir is None:
            raise ValueError("--image-dir is required with --labels")
        for image in cookies["images"]:
            coco_id = int(image["id"])
            prediction_id = prediction_id_by_coco_id[coco_id]
            file_name = image.get("file_name_RGB") or image.get("file_name")
            image_path = args.image_dir / file_name
            with Image.open(image_path) as handle:
                width, height = handle.size
            image_sizes[prediction_id] = (width, height)
            label_path = args.labels / f"{Path(file_name).stem}.txt"
            if not label_path.is_file():
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) < 5:
                    continue
                cls, xc, yc, bw, bh = fields[:5]
                xc_f, yc_f, bw_f, bh_f = map(float, (xc, yc, bw, bh))
                ground_truth[prediction_id].append(
                    (
                        int(float(cls)),
                        [
                            (xc_f - bw_f / 2) * width,
                            (yc_f - bh_f / 2) * height,
                            (xc_f + bw_f / 2) * width,
                            (yc_f + bh_f / 2) * height,
                        ],
                    )
                )
    else:
        for annotation in cookies["annotations"]:
            x, y, width, height = annotation["bbox"]
            image_id = prediction_id_by_coco_id[int(annotation["image_id"])]
            ground_truth[image_id].append(
                (
                    int(annotation["category_id"]),
                    [x, y, x + width, y + height],
                )
            )

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = defaultdict(list)
    for prediction in predictions:
        by_image[int(prediction["image_id"])].append(prediction)

    group_for_image: dict[int, str] = {}
    if args.groups_manifest:
        groups = json.loads(args.groups_manifest.read_text(encoding="utf-8"))
        stem_to_group = {
            stem: group_name
            for group_name, group in groups["groups"].items()
            for stem in group["stems"]
        }
        for image in cookies["images"]:
            stem = Path(
                image.get("file_name_RGB") or image.get("file_name") or ""
            ).stem
            group_for_image[prediction_id_by_coco_id[int(image["id"])]] = (
                stem_to_group[stem]
            )

    tp_rows, confidence_rows, class_rows, target_rows = [], [], [], []
    image_rows = []
    for image in cookies["images"]:
        image_id = prediction_id_by_coco_id[int(image["id"])]
        gt = ground_truth[image_id]
        gt_boxes = np.asarray([box for _, box in gt], dtype=np.float32).reshape(-1, 4)
        gt_classes = np.asarray([label for label, _ in gt], dtype=np.int64)
        image_predictions = by_image[image_id]
        pred_boxes = []
        pred_classes = []
        confidences = []
        for prediction in image_predictions:
            x, y, width, height = prediction["bbox"]
            pred_boxes.append([x, y, x + width, y + height])
            pred_classes.append(int(prediction["category_id"]) - 1)
            confidences.append(float(prediction["score"]))
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
        pred_classes = np.asarray(pred_classes, dtype=np.int64)
        confidences = np.asarray(confidences, dtype=np.float32)
        tp_row = match(pred_boxes, pred_classes, gt_boxes, gt_classes)
        tp_rows.append(tp_row)
        confidence_rows.append(confidences)
        class_rows.append(pred_classes)
        target_rows.append(gt_classes)
        image_rows.append(
            {
                "image_id": image_id,
                "tp": tp_row,
                "confidence": confidences,
                "pred_classes": pred_classes,
                "target_classes": gt_classes,
            }
        )

    tp = np.concatenate(tp_rows, axis=0)
    confidence = np.concatenate(confidence_rows, axis=0)
    pred_classes = np.concatenate(class_rows, axis=0)
    target_classes = np.concatenate(target_rows, axis=0)
    ap50, ap50_95 = ap_per_class(tp, confidence, pred_classes, target_classes)
    per_class = {}
    for class_id in sorted(np.unique(target_classes).tolist()):
        pred_keep = pred_classes == class_id
        target_keep = target_classes == class_id
        class_ap50, class_ap50_95 = ap_per_class(
            tp[pred_keep],
            confidence[pred_keep],
            pred_classes[pred_keep],
            target_classes[target_keep],
        )
        per_class[str(class_id)] = {
            "ap50": float(class_ap50),
            "ap50_95": float(class_ap50_95),
            "n_ground_truth_boxes": int(target_keep.sum()),
        }
    group_metrics = {}
    for group_name in sorted(set(group_for_image.values())):
        selected = [
            row for row in image_rows if group_for_image[row["image_id"]] == group_name
        ]
        group_tp = np.concatenate([row["tp"] for row in selected], axis=0)
        group_confidence = np.concatenate(
            [row["confidence"] for row in selected], axis=0
        )
        group_pred_classes = np.concatenate(
            [row["pred_classes"] for row in selected], axis=0
        )
        group_target_classes = np.concatenate(
            [row["target_classes"] for row in selected], axis=0
        )
        group_ap50, group_ap50_95 = ap_per_class(
            group_tp,
            group_confidence,
            group_pred_classes,
            group_target_classes,
        )
        group_metrics[group_name] = {
            "n_images": len(selected),
            "ap50": float(group_ap50),
            "ap50_95": float(group_ap50_95),
        }
    payload = {
        "predictions": str(args.predictions.resolve()),
        "annotations": str(args.annotations.resolve()),
        "n_images": len(cookies["images"]),
        "n_ground_truth_boxes": int(len(target_classes)),
        "n_predictions": int(len(confidence)),
        "metric": {
            "ap50": float(ap50),
            "ap50_95": float(ap50_95),
            "per_class": per_class,
            "groups": group_metrics,
        },
        "protocol": {
            "iou_thresholds": IOU_THRESHOLDS.tolist(),
            "matching": "maximum-cardinality one-to-one matching",
            "pooling": "detections pooled within the test split",
            "precision_recall_integration": "101-point interpolated AP",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
