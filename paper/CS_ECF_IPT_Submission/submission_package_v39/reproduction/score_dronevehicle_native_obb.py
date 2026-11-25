"""Score DroneVehicle candidates with native oriented-box IoU."""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torchvision_cuda_compat  # noqa: F401,E402
from local_ap_reference import ap_per_class  # noqa: E402


IOU_T = np.linspace(0.5, 0.95, 10)
CLASSES = ("car", "truck", "bus", "van", "freight car")
CDF_NAME = "consensus_intersection_pixel_mean_ifcnn_q095_a020_s05"


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(directory / stem)


def load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    polygons = []
    classes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name == "feright car":
            name = "freight car"
        if name not in CLASSES:
            continue
        polygon = obj.find("polygon")
        if polygon is None:
            continue
        points = [
            [
                float(polygon.findtext(f"x{index}")),
                float(polygon.findtext(f"y{index}")),
            ]
            for index in range(1, 5)
        ]
        polygons.append(points)
        classes.append(CLASSES.index(name))
    return (
        np.asarray(polygons, dtype=np.float32)
        if polygons
        else np.zeros((0, 4, 2), dtype=np.float32),
        np.asarray(classes, dtype=np.int64) if classes else np.zeros(
            (0,),
            dtype=np.int64,
        ),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--cdf-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    stems = [str(stem) for stem in manifest["stems"]["test"]]
    if args.limit:
        stems = stems[: args.limit]
    candidates = {
        "vis": args.sources / "visible",
        "cdf": args.cdf_dir,
        "pia": args.candidates / "sota_candidates" / "piafusion",
        "spg": args.candidates / "sota_candidates" / "spgfusion",
    }
    ground_truth = {
        stem: load_ground_truth(args.sources / "xml" / f"{stem}.xml")
        for stem in stems
    }
    args.out.mkdir(parents=True, exist_ok=True)

    for specification in args.model:
        name, path = specification.split("=", 1)
        model = YOLO(path)
        summary = {}
        for candidate, directory in candidates.items():
            paths = [find_image(directory, stem) for stem in stems]
            rows = []
            started = time.time()
            for offset in range(0, len(paths), args.batch_size):
                batch_paths = paths[offset:offset + args.batch_size]
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
                    rows.append((
                        match(
                            pred_polygons,
                            pred_classes,
                            gt_polygons,
                            gt_classes,
                        ),
                        confidence,
                        pred_classes,
                        gt_classes,
                    ))
            tp = np.concatenate([row[0] for row in rows], axis=0)
            conf = np.concatenate([row[1] for row in rows], axis=0)
            pred_cls = np.concatenate([row[2] for row in rows], axis=0)
            gt_cls = np.concatenate([row[3] for row in rows], axis=0)
            mean_50, mean_95 = ap_per_class(
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
                img_idx=np.concatenate([
                    np.full(len(row[1]), index, np.int32)
                    for index, row in enumerate(rows)
                ]),
                tcls=gt_cls,
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
