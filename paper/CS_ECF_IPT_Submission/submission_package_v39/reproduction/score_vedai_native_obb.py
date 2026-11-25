"""Score VEDAI candidates with native rotated-box IoU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torchvision_cuda_compat  # noqa: F401,E402
from local_ap_reference import compute_ap  # noqa: E402


IOU_T = np.linspace(0.5, 0.95, 10)
CANDIDATES = {
    "vis": "visible",
    "cdf": None,
    "pia": "piafusion",
    "spg": "spgfusion",
}


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(directory / stem)


def load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    polygons = []
    classes = []
    if path.is_file():
        for line in path.read_text().splitlines():
            values = line.split()
            if len(values) != 9:
                continue
            class_id = int(values[0])
            coords = np.asarray(values[1:], dtype=np.float32)
            coords = coords.reshape(4, 2) * 512.0
            polygons.append(coords)
            classes.append(class_id)
    return (
        np.asarray(polygons, dtype=np.float32)
        if polygons
        else np.zeros((0, 4, 2), dtype=np.float32),
        np.asarray(classes, dtype=np.int64)
        if classes
        else np.zeros((0,), dtype=np.int64),
    )


def match(
    pred_polygons: np.ndarray,
    pred_classes: np.ndarray,
    gt_polygons: np.ndarray,
    gt_classes: np.ndarray,
) -> np.ndarray:
    correct = np.zeros((len(pred_polygons), len(IOU_T)), dtype=bool)
    if len(pred_polygons) == 0 or len(gt_polygons) == 0:
        return correct
    iou = batch_probiou(
        xyxyxyxy2xywhr(torch.from_numpy(pred_polygons)),
        xyxyxyxy2xywhr(torch.from_numpy(gt_polygons)),
    ).numpy()
    class_matches = pred_classes[:, None] == gt_classes[None, :]
    for index, threshold in enumerate(IOU_T):
        valid = np.logical_and(iou >= threshold, class_matches)
        if valid.any():
            rows, columns = linear_sum_assignment(np.logical_not(valid))
            correct[rows, index] = valid[rows, columns]
    return correct


def complete_ap(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
) -> tuple[float, float]:
    if len(target_cls) == 0:
        return 0.0, 0.0
    order = np.argsort(-conf)
    tp = tp[order]
    pred_cls = pred_cls[order]
    class_aps = []
    for class_id, count in zip(*np.unique(target_cls, return_counts=True)):
        selected = pred_cls == class_id
        if not selected.any():
            class_aps.append(np.zeros(len(IOU_T)))
            continue
        tpc = np.cumsum(tp[selected], axis=0)
        fpc = np.cumsum(1 - tp[selected], axis=0)
        recall = tpc / max(float(count), 1.0)
        precision = tpc / np.maximum(tpc + fpc, 1e-16)
        class_aps.append(
            np.asarray(
                [
                    compute_ap(recall[:, index], precision[:, index])
                    for index in range(len(IOU_T))
                ]
            )
        )
    values = np.asarray(class_aps)
    return float(values[:, 0].mean()), float(values.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--cdf-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    stems = sorted(path.stem for path in args.labels.glob("*.txt"))
    if args.limit:
        stems = stems[: args.limit]
    directories = {
        "vis": args.sources / "visible",
        "cdf": args.cdf_dir,
        "pia": args.candidates / "piafusion",
        "spg": args.candidates / "spgfusion",
    }
    ground_truth = {
        stem: load_ground_truth(args.labels / f"{stem}.txt")
        for stem in stems
    }
    args.out.mkdir(parents=True, exist_ok=True)

    for specification in args.model:
        name, path = specification.split("=", 1)
        model = YOLO(path)
        summary = {}
        for candidate, directory in directories.items():
            paths = [find_image(directory, stem) for stem in stems]
            rows = []
            started = time.time()
            for offset in range(0, len(paths), args.batch_size):
                batch_paths = paths[offset : offset + args.batch_size]
                predictions = model.predict(
                    [str(path) for path in batch_paths],
                    conf=0.001,
                    imgsz=640,
                    device=0,
                    verbose=False,
                )
                for local, prediction in enumerate(predictions):
                    stem = stems[offset + local]
                    obb = prediction.obb
                    pred_polygons = obb.xyxyxyxy.cpu().numpy().astype(
                        np.float32
                    )
                    pred_classes = obb.cls.cpu().numpy().astype(np.int64)
                    confidence = obb.conf.cpu().numpy().astype(np.float32)
                    gt_polygons, gt_classes = ground_truth[stem]
                    rows.append(
                        (
                            match(
                                pred_polygons,
                                pred_classes,
                                gt_polygons,
                                gt_classes,
                            ),
                            confidence,
                            pred_classes,
                            gt_classes,
                            pred_polygons,
                        )
                    )
            tp = np.concatenate([row[0] for row in rows], axis=0)
            conf = np.concatenate([row[1] for row in rows])
            pred_cls = np.concatenate([row[2] for row in rows])
            gt_cls = np.concatenate([row[3] for row in rows])
            mean_50, mean_95 = complete_ap(
                tp,
                conf,
                pred_cls,
                gt_cls,
            )
            np.savez_compressed(
                args.out / f"{name}__{candidate}.npz",
                tp=tp,
                conf=conf,
                pcls=pred_cls,
                img_idx=np.concatenate(
                    [
                        np.full(len(row[1]), index, np.int32)
                        for index, row in enumerate(rows)
                    ]
                ),
                tcls=gt_cls,
                polygons=np.concatenate([row[4] for row in rows]),
                poly_img_idx=np.concatenate(
                    [
                        np.full(len(row[1]), index, np.int32)
                        for index, row in enumerate(rows)
                    ]
                ),
                gt_polygons=np.concatenate(
                    [ground_truth[stem][0] for stem in stems]
                ),
                gt_img_idx=np.concatenate(
                    [
                        np.full(
                            len(ground_truth[stem][0]),
                            index,
                            np.int32,
                        )
                        for index, stem in enumerate(stems)
                    ]
                ),
                stems=np.asarray(stems),
                n_img=len(stems),
                n_gt=np.asarray(
                    [len(row[3]) for row in rows],
                    dtype=np.int32,
                ),
            )
            summary[candidate] = {
                "mAP50": mean_50,
                "mAP50_95": mean_95,
                "seconds": time.time() - started,
            }
            print(
                f"{name} {candidate} mAP50={mean_50:.4f} "
                f"mAP50-95={mean_95:.4f}",
                flush=True,
            )
        (args.out / f"{name}_scores.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
