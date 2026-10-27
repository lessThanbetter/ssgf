"""Score drone vehicle images with the fixed YOLO11x-OBB detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, r"D:\Code\_server_handoff_v2\ssgf\scripts")
import torchvision_cuda_compat  # noqa: F401,E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def load_ground_truth(path: Path, width: int, height: int):
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            rows.append([float(value) for value in fields[1:5]])
    boxes = np.asarray(rows, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height
    centers_x = boxes[:, 0]
    centers_y = boxes[:, 1]
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    return np.stack(
        (
            centers_x - widths / 2,
            centers_y - heights / 2,
            centers_x + widths / 2,
            centers_y + heights / 2,
        ),
        axis=1,
    ).astype(np.float32)


def match_flags(gt_boxes, pred_boxes, confidences):
    n_detections = len(confidences)
    flags = np.zeros((n_detections, len(IOU_THRESHOLDS)), dtype=bool)
    if n_detections == 0 or len(gt_boxes) == 0:
        return flags
    iou = box_iou(
        torch.as_tensor(gt_boxes, dtype=torch.float32),
        torch.as_tensor(pred_boxes, dtype=torch.float32),
    ).cpu().numpy()
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
        working = iou.copy()
        matched_gt = np.full(len(gt_boxes), -1, dtype=np.int64)
        candidates = np.flatnonzero((working > threshold).any(axis=0))
        order = candidates[
            np.argsort(-np.asarray(confidences)[candidates], kind="stable")
        ]
        for detection_index in order:
            gt_index = int(working[:, detection_index].argmax())
            if working[gt_index, detection_index] > threshold:
                matched_gt[gt_index] = detection_index
                working[gt_index] = 0
        matched = matched_gt[matched_gt >= 0]
        flags[matched, threshold_index] = True
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    image_paths = sorted(
        path
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for path in args.images.glob(suffix)
    )
    if args.limit:
        image_paths = image_paths[: args.limit]

    model = YOLO(str(args.model))
    tp_rows, confidence_rows, predicted_class_rows = [], [], []
    target_class_rows, n_gt_rows, image_index_rows = [], [], []

    for image_index, image_path in enumerate(image_paths):
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=0.001,
            iou=0.7,
            batch=1,
            stream=False,
            verbose=False,
        )[0]
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        ground_truth = load_ground_truth(
            args.labels / f"{image_path.stem}.txt", width, height
        )
        if result.obb is None or len(result.obb) == 0:
            predicted = np.zeros((0, 5), dtype=np.float32)
            confidence = np.zeros(0, dtype=np.float32)
        else:
            predicted = result.obb.xyxy.cpu().numpy().astype(np.float32)
            confidence = result.obb.conf.cpu().numpy().astype(np.float32)
        flags = match_flags(ground_truth, predicted, confidence)
        tp_rows.append(flags)
        confidence_rows.append(confidence)
        predicted_class_rows.append(np.zeros(len(confidence), dtype=np.float32))
        target_class_rows.append(
            np.zeros(len(ground_truth), dtype=np.float32)
        )
        n_gt_rows.append(len(ground_truth))
        image_index_rows.append(
            np.full(len(confidence), image_index, dtype=np.int64)
        )
        if (image_index + 1) % 25 == 0 or image_index + 1 == len(image_paths):
            print(f"{image_index + 1}/{len(image_paths)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        tp=np.concatenate(tp_rows, axis=0)
        if tp_rows
        else np.zeros((0, 10), dtype=bool),
        conf=np.concatenate(confidence_rows)
        if confidence_rows
        else np.zeros(0, dtype=np.float32),
        pcls=np.concatenate(predicted_class_rows)
        if predicted_class_rows
        else np.zeros(0, dtype=np.float32),
        tcls=np.concatenate(target_class_rows)
        if target_class_rows
        else np.zeros(0, dtype=np.float32),
        n_gt=np.asarray(n_gt_rows, dtype=np.int64),
        img_idx=np.concatenate(image_index_rows)
        if image_index_rows
        else np.zeros(0, dtype=np.int64),
        stems=np.asarray([path.stem for path in image_paths]),
        n_img=len(image_paths),
    )
    print(
        json.dumps(
            {
                "images": len(image_paths),
                "detections": int(sum(map(len, confidence_rows))),
                "ground_truth": int(sum(n_gt_rows)),
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
