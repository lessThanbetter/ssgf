"""Rescore LLVIP confirmation data with confidence-ordered matching."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(HERE))

import torchvision_cuda_compat  # noqa: E402,F401
from ultralytics import RTDETR, YOLO  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

from local_ap_reference import load as load_per_image  # noqa: E402
from local_ap_reference import pooled  # noqa: E402


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
SEQUENCES = ("050", "060", "070")
MODEL_SPECS = {
    "yolo26_s0": (
        "yolo",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\yolo26n_clean\weights\last.pt"
        ),
    ),
    "yolo26_s1": (
        "yolo",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\yolo26n_clean_s1\weights\last.pt"
        ),
    ),
    "yolo26_s2": (
        "yolo",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\yolo26n_clean_s2\weights\last.pt"
        ),
    ),
    "yolo8_s0": (
        "yolo",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\yolo8n_clean_holdout\weights\last.pt"
        ),
    ),
    "yolo8_s1": (
        "yolo",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\yolo8n_clean_holdout_s1\weights\last.pt"
        ),
    ),
    "rtdetr_l": (
        "rtdetr",
        Path(
            r"D:\Code\_ssgf_external\r2_clean\runs"
            r"\rtdetr_l_clean\weights\last.pt"
        ),
    ),
}
CANDIDATE_DIRS = {
    "vis": None,
    "cdf": "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05",
    "piafusion": "piafusion",
    "spgfusion": "spgfusion",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_image(directory: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def load_ground_truth(
    label_dir: Path,
    stem: str,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    classes = []
    label_path = label_dir / f"{stem}.txt"
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            xc, yc, width_fraction, height_fraction = map(
                float,
                parts[1:5],
            )
            boxes.append(
                [
                    (xc - width_fraction / 2.0) * width,
                    (yc - height_fraction / 2.0) * height,
                    (xc + width_fraction / 2.0) * width,
                    (yc + height_fraction / 2.0) * height,
                ]
            )
            classes.append(class_id)
    return (
        np.asarray(boxes, dtype=np.float32)
        if boxes
        else np.zeros((0, 4), dtype=np.float32),
        np.asarray(classes, dtype=np.int64)
        if classes
        else np.zeros((0,), dtype=np.int64),
    )


def confidence_ordered_match(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_confidence: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> np.ndarray:
    """Match each detection in descending confidence order."""
    correct = np.zeros(
        (len(pred_boxes), len(IOU_THRESHOLDS)),
        dtype=bool,
    )
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return correct

    overlaps = box_iou(
        torch.from_numpy(pred_boxes),
        torch.from_numpy(gt_boxes),
    ).numpy()
    same_class = pred_classes[:, None] == gt_classes[None, :]
    confidence_order = np.argsort(-pred_confidence, kind="stable")
    for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
        unmatched = set(range(len(gt_boxes)))
        for prediction_index in confidence_order:
            eligible = [
                ground_truth_index
                for ground_truth_index in unmatched
                if same_class[prediction_index, ground_truth_index]
                and overlaps[prediction_index, ground_truth_index]
                >= threshold
            ]
            if not eligible:
                continue
            ground_truth_index = max(
                eligible,
                key=lambda index: overlaps[prediction_index, index],
            )
            unmatched.remove(ground_truth_index)
            correct[prediction_index, threshold_index] = True
    return correct


def candidate_directory(
    candidate_root: Path,
    sequence: str,
    candidate: str,
) -> Path:
    directory_name = CANDIDATE_DIRS[candidate]
    if directory_name is None:
        return (
            PACKAGE.parent.parent.parent
            / "data"
            / f"erf_holdout_llvip{sequence}"
            / "visible"
        )
    return (
        candidate_root
        / f"llvip{sequence}"
        / "candidates"
        / directory_name
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=PACKAGE / "evidence" / "prospective",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\Code\_server_handoff_v2\ssgf\data"),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(
            r"D:\Code\_server_handoff_v2\ssgf\results"
            r"\prospective_20261029"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PACKAGE / "evidence" / "prospective_standard_matching",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=SEQUENCES,
        default=list(SEQUENCES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS),
    )
    args = parser.parse_args()

    reference_stems = {}
    for sequence in SEQUENCES:
        reference = np.load(
            args.reference_root
            / f"llvip{sequence}"
            / "scores"
            / "yolo26_s0__vis.npz",
            allow_pickle=True,
        )
        stems = [
            str(value) for value in reference["stems"]
        ]
        if args.limit is not None:
            stems = stems[: args.limit]
        reference_stems[sequence] = stems

    metadata = {
        "method": "confidence-ordered one-to-one matching",
        "matching_rule": (
            "At each IoU threshold, detections are traversed in descending "
            "confidence order. Each detection is assigned to the unmatched "
            "same-class ground-truth box with maximum IoU above threshold."
        ),
        "iou_thresholds": IOU_THRESHOLDS.tolist(),
        "confidence_floor": 0.001,
        "nms_iou": 0.7,
        "max_detections": 300,
        "image_size": 640,
        "sequences": list(args.sequences),
        "limit": args.limit,
        "models": {},
    }

    for model_name in args.models:
        model_kind, model_path = MODEL_SPECS[model_name]
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        model = (
            RTDETR(str(model_path))
            if model_kind == "rtdetr"
            else YOLO(str(model_path))
        )
        metadata["models"][model_name] = {
            "kind": model_kind,
            "path": str(model_path),
            "sha256": sha256(model_path),
        }
        for sequence in args.sequences:
            stems = reference_stems[sequence]
            label_dir = (
                args.data_root
                / f"erf_holdout_llvip{sequence}"
                / "labels"
            )
            output_dir = args.out_root / f"llvip{sequence}" / "scores"
            output_dir.mkdir(parents=True, exist_ok=True)

            for candidate in ("vis", "cdf", "piafusion", "spgfusion"):
                image_dir = candidate_directory(
                    args.candidate_root,
                    sequence,
                    candidate,
                )
                paths = []
                for stem in stems:
                    path = find_image(image_dir, stem)
                    if path is None:
                        raise FileNotFoundError(
                            f"{sequence}/{candidate}: missing {stem}"
                        )
                    paths.append(path)

                started = time.time()
                rows = []
                for offset in range(0, len(paths), args.batch):
                    batch_paths = paths[offset : offset + args.batch]
                    results = model.predict(
                        [str(path) for path in batch_paths],
                        conf=0.001,
                        iou=0.7,
                        imgsz=640,
                        max_det=300,
                        device=args.device,
                        verbose=False,
                    )
                    if len(results) != len(batch_paths):
                        raise RuntimeError(
                            f"{model_name}/{sequence}/{candidate}: "
                            f"{len(results)} results for {len(batch_paths)} "
                            "images"
                        )
                    for result, image_path, stem in zip(
                        results,
                        batch_paths,
                        stems[offset : offset + args.batch],
                    ):
                        with Image.open(image_path) as image:
                            width, height = image.size
                        gt_boxes, gt_classes = load_ground_truth(
                            label_dir,
                            stem,
                            width,
                            height,
                        )
                        pred_boxes = (
                            result.boxes.xyxy.cpu().numpy().astype(np.float32)
                        )
                        pred_classes = (
                            result.boxes.cls.cpu().numpy().astype(np.int64)
                        )
                        confidence = (
                            result.boxes.conf.cpu().numpy().astype(np.float32)
                        )
                        rows.append(
                            (
                                confidence_ordered_match(
                                    pred_boxes,
                                    pred_classes,
                                    confidence,
                                    gt_boxes,
                                    gt_classes,
                                ),
                                confidence,
                                pred_classes,
                                gt_classes,
                            )
                        )

                tp = np.concatenate([row[0] for row in rows], axis=0)
                confidence = np.concatenate([row[1] for row in rows])
                pred_classes = np.concatenate([row[2] for row in rows])
                target_classes = np.concatenate([row[3] for row in rows])
                image_index = np.concatenate([
                    np.full(len(row[1]), index, dtype=np.int32)
                    for index, row in enumerate(rows)
                ])
                output_path = output_dir / f"{model_name}__{candidate}.npz"
                np.savez_compressed(
                    output_path,
                    tp=tp,
                    conf=confidence,
                    pcls=pred_classes,
                    img_idx=image_index,
                    tcls=target_classes,
                    stems=np.asarray(stems),
                    n_img=len(stems),
                    n_gt=np.asarray(
                        [len(row[3]) for row in rows],
                        dtype=np.int32,
                    ),
                )
                ap50, ap50_95 = pooled(
                    load_per_image(str(output_path)),
                    np.arange(len(stems)),
                )
                print(
                    f"{model_name:10s} {sequence} {candidate:3s} "
                    f"AP50={ap50:.6f} AP50-95={ap50_95:.6f} "
                    f"{time.time() - started:.1f}s",
                    flush=True,
                )

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print("WROTE", args.out_root)


if __name__ == "__main__":
    main()
