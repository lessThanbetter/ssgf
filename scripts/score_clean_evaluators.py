"""Evaluate clean-split detectors on every LLVIP input and bootstrap contrasts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import RTDETR, YOLO
from ultralytics.utils.metrics import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torchvision_cuda_compat  # noqa: E402,F401
from local_ap_reference import (  # noqa: E402
    ap_per_class,
    load as load_per_image,
    pooled as pooled_per_image,
)


IOU_T = np.linspace(0.5, 0.95, 10)
INPUTS = [
    "vis",
    "ir",
    "mean",
    "activity_only",
    "tophat",
    "probe",
    "tophat_permuted",
    "probe_permuted",
    "tophat_wshuffle",
    "probe_wshuffle",
    "ensemble",
    "ifcnn",
    "densefuse",
    "csf",
    "piafusion",
    "spgfusion",
]


def image_dir(kind: str, root: Path) -> Path:
    if kind in {"vis", "ir"}:
        return root / "test" / ("visible" if kind == "vis" else "infrared")
    if kind in {
        "mean",
        "activity_only",
        "tophat",
        "probe",
        "tophat_permuted",
        "probe_permuted",
        "tophat_wshuffle",
        "probe_wshuffle",
    }:
        return root / "k1_native" / "imgs" / kind
    if kind in {"ensemble", "ifcnn", "densefuse"}:
        return root / "eval_fused" / kind / "images"
    return root / "sota_outs" / {
        "csf": "csf_llvip_native",
        "piafusion": "pia_llvip_native_fp16",
        "spgfusion": "spg_llvip_native_fp16",
    }[kind]


def find_image(directory: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def load_ground_truth(label_dir: Path, stem: str, width: int, height: int):
    boxes, classes = [], []
    path = label_dir / f"{stem}.txt"
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
    iou = box_iou(
        torch.from_numpy(pred_boxes),
        torch.from_numpy(gt_boxes),
    ).numpy()
    class_matches = pred_classes[:, None] == gt_classes[None, :]
    for index, threshold in enumerate(IOU_T):
        valid = np.logical_and(iou >= threshold, class_matches)
        if valid.any():
            rows, columns = linear_sum_assignment(np.logical_not(valid))
            correct[rows, index] = valid[rows, columns]
    return correct


def pooled(rows, indices=None):
    selected = rows if indices is None else [rows[i] for i in indices]
    selected = [row for row in selected if row is not None]
    if not selected:
        return 0.0, 0.0
    tp = np.concatenate([row[0] for row in selected], axis=0)
    conf = np.concatenate([row[1] for row in selected], axis=0)
    pred_cls = np.concatenate([row[2] for row in selected], axis=0)
    target_cls = np.concatenate([row[3] for row in selected], axis=0)
    return ap_per_class(tp, conf, pred_cls, target_cls)


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    """Return BH adjusted q values, preserving the input keys."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank in range(count, 0, -1):
        key, p_value = ordered[rank - 1]
        running = min(running, p_value * count / rank)
        adjusted[key] = min(1.0, running)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--stems-npz", required=True)
    parser.add_argument("--weights", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inputs", nargs="+", default=INPUTS)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    root = Path(args.root)
    label_dir = Path(args.labels)
    output = Path(args.out)
    per_image = output / "per_image"
    per_image.mkdir(parents=True, exist_ok=True)

    reference = np.load(args.stems_npz, allow_pickle=True)
    stems = [str(stem) for stem in reference["stems"]]
    if args.limit:
        stems = stems[:args.limit]
    n_images = len(stems)
    first_image = find_image(image_dir("vis", root), stems[0])
    if first_image is None:
        raise SystemExit("visible test image missing")
    from PIL import Image

    with Image.open(first_image) as image:
        width, height = image.size
    ground_truth = {
        stem: load_ground_truth(label_dir, stem, width, height) for stem in stems
    }

    summary = {}
    multiplicity = {}
    rng = np.random.default_rng(0)
    draws = [
        rng.integers(0, n_images, size=n_images)
        for _ in range(args.bootstrap)
    ]
    for specification in args.weights:
        name, kind, path = specification.split("=", 2)
        model = RTDETR(path) if kind == "rtdetr" else YOLO(path)
        pools = {}
        summary[name] = {}
        for input_name in args.inputs:
            directory = image_dir(input_name, root)
            rows = []
            paths = []
            for stem in stems:
                image_path = find_image(directory, stem)
                if image_path is None:
                    raise SystemExit(f"{input_name}: missing image for {stem}")
                paths.append(image_path)
            start = time.time()
            batch_size = max(1, args.batch_size)
            for offset in range(0, n_images, batch_size):
                batch_paths = paths[offset : offset + batch_size]
                results = model.predict(
                    [str(batch_path) for batch_path in batch_paths],
                    conf=0.001,
                    imgsz=640,
                    device=0,
                    verbose=False,
                )
                if len(results) != len(batch_paths):
                    raise RuntimeError(
                        f"{input_name}: {len(results)} results for "
                        f"{len(batch_paths)} images"
                    )
                for batch_index, result in enumerate(results):
                    stem = stems[offset + batch_index]
                    pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                    pred_classes = result.boxes.cls.cpu().numpy().astype(np.int64)
                    conf = result.boxes.conf.cpu().numpy().astype(np.float32)
                    gt_boxes, gt_classes = ground_truth[stem]
                    rows.append(
                        (
                            match(
                                pred_boxes,
                                pred_classes,
                                gt_boxes,
                                gt_classes,
                            ),
                            conf,
                            pred_classes,
                            gt_classes,
                        )
                    )
            pools[input_name] = rows
            m50, m95 = pooled(rows)
            summary[name][input_name] = {
                "mAP50": round(m50, 4),
                "mAP50_95": round(m95, 4),
            }
            np.savez_compressed(
                per_image / f"{name}__{input_name}.npz",
                tp=np.concatenate([row[0] for row in rows], axis=0),
                conf=np.concatenate([row[1] for row in rows], axis=0),
                pcls=np.concatenate([row[2] for row in rows], axis=0),
                img_idx=np.concatenate(
                    [np.full(len(row[1]), i, np.int32) for i, row in enumerate(rows)]
                ),
                tcls=np.concatenate([row[3] for row in rows], axis=0),
                stems=np.array(stems),
                n_img=n_images,
                n_gt=np.asarray([len(row[3]) for row in rows], np.int32),
            )
            print(
                f"{name:14s} {input_name:18s} "
                f"mAP50={m50:.4f} mAP50-95={m95:.4f} "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )

        analysis_pools = {
            input_name: load_per_image(
                str(per_image / f"{name}__{input_name}.npz")
            )
            for input_name in args.inputs
        }
        base50 = np.asarray(
            [pooled_per_image(analysis_pools["vis"], draw)[0] for draw in draws]
        )
        base95 = np.asarray(
            [pooled_per_image(analysis_pools["vis"], draw)[1] for draw in draws]
        )
        vis50, vis95 = pooled_per_image(analysis_pools["vis"], np.arange(n_images))
        p_values50: dict[str, float] = {}
        for input_name in args.inputs:
            if input_name == "vis":
                continue
            values50 = np.asarray(
                [
                    pooled_per_image(analysis_pools[input_name], draw)[0]
                    for draw in draws
                ]
            )
            values95 = np.asarray(
                [
                    pooled_per_image(analysis_pools[input_name], draw)[1]
                    for draw in draws
                ]
            )
            delta50 = values50 - base50
            delta95 = values95 - base95
            point50, point95 = pooled_per_image(
                analysis_pools[input_name],
                np.arange(n_images),
            )
            observed50 = point50 - vis50
            observed95 = point95 - vis95
            p_value50 = min(
                1.0,
                2.0
                * min(
                    (1 + int(np.sum(delta50 >= 0))) / (args.bootstrap + 1),
                    (1 + int(np.sum(delta50 <= 0))) / (args.bootstrap + 1),
                ),
            )
            if input_name != "ir":
                p_values50[input_name] = p_value50
            summary[name][input_name].update(
                {
                    "delta50_observed": round(float(observed50), 4),
                    "delta95_observed": round(float(observed95), 4),
                    "delta50": round(float(delta50.mean()), 4),
                    "lo50": round(float(np.percentile(delta50, 2.5)), 4),
                    "hi50": round(float(np.percentile(delta50, 97.5)), 4),
                    "delta95": round(float(delta95.mean()), 4),
                    "lo95": round(float(np.percentile(delta95, 2.5)), 4),
                    "hi95": round(float(np.percentile(delta95, 97.5)), 4),
                    "p_boot50": round(float(p_value50), 6),
                }
            )
        q_values50 = benjamini_hochberg(p_values50)
        multiplicity[name] = {
            "family": "fourteen fused inputs against the colour-scored visible source",
            "correction": "Benjamini-Hochberg",
            "alpha": 0.05,
            "n_tests": len(p_values50),
            "tests": {
                input_name: {
                    "delta50_observed": summary[name][input_name][
                        "delta50_observed"
                    ],
                    "p_boot50": summary[name][input_name]["p_boot50"],
                    "q_bh50": round(float(q_values50[input_name]), 6),
                }
                for input_name in p_values50
            },
        }
        (output / "scores.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (output / "multiplicity.json").write_text(
            json.dumps(multiplicity, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
