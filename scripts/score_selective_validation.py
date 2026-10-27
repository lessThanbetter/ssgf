"""Score validation candidates with detectors that never saw the validation pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from ultralytics import RTDETR, YOLO
from ultralytics.utils.metrics import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torchvision_cuda_compat  # noqa: F401,E402
from local_ap_reference import ap_per_class  # noqa: E402


IOU_T = np.linspace(0.5, 0.95, 10)


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"{directory / stem}.*")


def image_size(directory: Path, stem: str) -> tuple[int, int]:
    path = find_image(directory, stem)
    with Image.open(path) as image:
        return image.size


def load_ground_truth(path: Path, width: int, height: int):
    boxes, classes = [], []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, x, y, w, h = int(float(parts[0])), *map(float, parts[1:5])
            boxes.append(
                [
                    (x - w / 2) * width,
                    (y - h / 2) * height,
                    (x + w / 2) * width,
                    (y + h / 2) * height,
                ]
            )
            classes.append(cls)
    return (
        np.asarray(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32),
        np.asarray(classes, np.int64) if classes else np.zeros((0,), np.int64),
    )


def match(pred_boxes, pred_classes, gt_boxes, gt_classes):
    correct = np.zeros((len(pred_boxes), len(IOU_T)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return correct
    iou = box_iou(torch.from_numpy(pred_boxes), torch.from_numpy(gt_boxes)).numpy()
    class_matches = pred_classes[:, None] == gt_classes[None, :]
    for index, threshold in enumerate(IOU_T):
        valid = np.logical_and(iou >= threshold, class_matches)
        if valid.any():
            rows, columns = linear_sum_assignment(np.logical_not(valid))
            correct[rows, index] = valid[rows, columns]
    return correct


def pooled(rows):
    tp = np.concatenate([row[0] for row in rows], axis=0)
    confidence = np.concatenate([row[1] for row in rows], axis=0)
    pred_cls = np.concatenate([row[2] for row in rows], axis=0)
    target_cls = np.concatenate([row[3] for row in rows], axis=0)
    return ap_per_class(tp, confidence, pred_cls, target_cls)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="results/selective_fusion/validation_candidates")
    parser.add_argument("--visible", default="D:/Code/_ssgf_external/r2_clean/clean/images/val")
    parser.add_argument("--labels", default="D:/Code/_ssgf_external/r2_clean/clean/labels/val")
    parser.add_argument("--manifest", default="_deposit/ssgf-llvip-fusion-detection-v1/data/splits/clean_split_manifest.json")
    parser.add_argument(
        "--manifest-split",
        choices=("train", "validation", "test", "all"),
        default="validation",
    )
    parser.add_argument("--out", default="results/selective_fusion/validation_scores")
    parser.add_argument("--weights", action="append", required=True)
    parser.add_argument(
        "--candidate-names",
        default="",
        help="Comma-separated candidate names to score; empty scores every candidate.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.manifest_split == "all":
        values = manifest.get("stems")
        if values is None:
            raise ValueError("Manifest has no top-level 'stems' for split=all")
    else:
        if "split_stems" in manifest:
            values = manifest["split_stems"][args.manifest_split]
        else:
            stems_by_split = manifest.get("stems", {})
            key = args.manifest_split
            if key not in stems_by_split and key == "validation":
                key = "val"
            if key not in stems_by_split:
                raise KeyError(f"Manifest has no stems for split {key!r}")
            values = stems_by_split[key]
    stems = [str(value) for value in values]
    if args.limit:
        stems = stems[: args.limit]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    candidate_root = Path(args.candidates)
    candidates = {
        "vis": Path(args.visible),
        "mean": candidate_root / "mean",
        "activity_only": candidate_root / "activity_only",
        "tophat": candidate_root / "tophat",
        "probe": candidate_root / "probe",
        "ensemble": candidate_root / "ensemble",
        "ifcnn": candidate_root / "ifcnn",
        "densefuse": candidate_root / "densefuse",
        "csf": candidate_root / "csf",
    }
    for candidate_dir in sorted(candidate_root.iterdir()):
        if candidate_dir.is_dir():
            candidates.setdefault(candidate_dir.name, candidate_dir)
    if args.candidate_names:
        requested = [name.strip() for name in args.candidate_names.split(",") if name.strip()]
        missing = [name for name in requested if name not in candidates]
        if missing:
            raise ValueError(f"Unknown candidates: {', '.join(missing)}")
        candidates = {name: candidates[name] for name in requested}
    ground_truth = {}
    for stem in stems:
        width, height = image_size(Path(args.visible), stem)
        ground_truth[stem] = load_ground_truth(
            Path(args.labels) / f"{stem}.txt", width, height
        )
    summary = {}
    for specification in args.weights:
        name, kind, weights = specification.split("=", 2)
        model = RTDETR(weights) if kind == "rtdetr" else YOLO(weights)
        summary[name] = {}
        for candidate, directory in candidates.items():
            paths = [find_image(directory, stem) for stem in stems]
            rows = []
            start = time.time()
            for offset in range(0, len(paths), args.batch_size):
                batch_paths = paths[offset : offset + args.batch_size]
                results = model.predict(
                    [str(path) for path in batch_paths],
                    conf=0.001,
                    imgsz=640,
                    device=0,
                    verbose=False,
                )
                for local_index, result in enumerate(results):
                    stem = stems[offset + local_index]
                    prediction = getattr(result, "obb", None)
                    if prediction is None:
                        prediction = result.boxes
                    pred_boxes = prediction.xyxy.cpu().numpy().astype(np.float32)
                    pred_classes = prediction.cls.cpu().numpy().astype(np.int64)
                    confidence = prediction.conf.cpu().numpy().astype(np.float32)
                    gt_boxes, gt_classes = ground_truth[stem]
                    rows.append(
                        (
                            match(pred_boxes, pred_classes, gt_boxes, gt_classes),
                            confidence,
                            pred_classes,
                            gt_classes,
                        )
                    )
            mean_50, mean_95 = pooled(rows)
            summary[name][candidate] = {
                "mAP50": round(mean_50, 4),
                "mAP50_95": round(mean_95, 4),
            }
            np.savez_compressed(
                output / f"{name}__{candidate}.npz",
                tp=np.concatenate([row[0] for row in rows], axis=0),
                conf=np.concatenate([row[1] for row in rows], axis=0),
                pcls=np.concatenate([row[2] for row in rows], axis=0),
                img_idx=np.concatenate(
                    [np.full(len(row[1]), index, np.int32) for index, row in enumerate(rows)]
                ),
                tcls=np.concatenate([row[3] for row in rows], axis=0),
                stems=np.asarray(stems),
                n_img=len(stems),
                n_gt=np.asarray([len(row[3]) for row in rows], np.int32),
            )
            print(
                f"{name:8s} {candidate:14s} mAP50={mean_50:.4f} "
                f"mAP50-95={mean_95:.4f} ({time.time() - start:.0f}s)",
                flush=True,
            )
    (output / "scores.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("WROTE", output / "scores.json")


if __name__ == "__main__":
    main()
