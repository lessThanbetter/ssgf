"""Score visible and PIAFusion inputs on the fixed prospective holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torchvision_cuda_compat  # noqa: F401
from ultralytics import RTDETR, YOLO

from local_ap_reference import load as load_per_image
from local_ap_reference import pooled
from rescore_prospective_standard_matching import (
    confidence_ordered_match,
    load_ground_truth,
)


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE = SCRIPT_PATH.parents[1]
REPO_ROOT = next(
    path
    for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
    if (path / "data").is_dir()
    and (path / "scripts").is_dir()
)
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
SEQUENCES = ("080", "090", "100")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "data" / "prospective_pia_sources",
    )
    parser.add_argument(
        "--pia-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "prospective_pia"
            / "candidates"
            / "piafusion"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            PACKAGE
            / "evidence"
            / "prospective_pia_confirmation"
            / "scores"
        ),
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    source_manifest = json.loads(
        (args.source_root / "manifest.json").read_text(encoding="utf-8")
    )
    stems_by_sequence = {}
    for sequence in args.sequences:
        stems = [
            str(stem)
            for stem in source_manifest["sequence_stems"][sequence]
        ]
        if args.limit is not None:
            stems = stems[: args.limit]
        stems_by_sequence[sequence] = stems
    metadata = {
        "method": "confidence-ordered one-to-one matching",
        "confidence_floor": 0.001,
        "nms_iou": 0.7,
        "max_detections": 300,
        "image_size": 640,
        "batch_size": args.batch,
        "sequences": list(args.sequences),
        "candidates": ["vis", "piafusion"],
        "models": {},
        "source_manifest_sha256": sha256(
            args.source_root / "manifest.json"
        ),
        "pia_files": 0,
    }

    for model_name in args.models:
        model_kind, model_path = MODEL_SPECS[model_name]
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
            stems = stems_by_sequence[sequence]
            for candidate in ("vis", "piafusion"):
                image_dir = (
                    args.source_root / "visible"
                    if candidate == "vis"
                    else args.pia_dir
                )
                paths = [
                    image_dir / f"{stem}{'.jpg' if candidate == 'vis' else '.png'}"
                    for stem in stems
                ]
                missing = [path for path in paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(missing[0])
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
                            f"result count mismatch"
                        )
                    for result, image_path, stem in zip(
                        results,
                        batch_paths,
                        stems[offset : offset + args.batch],
                    ):
                        with Image.open(image_path) as image:
                            width, height = image.size
                        gt_boxes, gt_classes = load_ground_truth(
                            args.source_root / "labels",
                            stem,
                            width,
                            height,
                        )
                        pred_boxes = result.boxes.xyxy.cpu().numpy().astype(
                            np.float32
                        )
                        pred_classes = result.boxes.cls.cpu().numpy().astype(
                            np.int64
                        )
                        confidence = result.boxes.conf.cpu().numpy().astype(
                            np.float32
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
                output_dir = args.out / f"llvip{sequence}"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{model_name}__{candidate}.npz"
                np.savez_compressed(
                    output_path,
                    tp=np.concatenate([row[0] for row in rows], axis=0),
                    conf=np.concatenate([row[1] for row in rows]),
                    pcls=np.concatenate([row[2] for row in rows]),
                    img_idx=np.concatenate(
                        [
                            np.full(len(row[1]), index, dtype=np.int32)
                            for index, row in enumerate(rows)
                        ]
                    ),
                    tcls=np.concatenate([row[3] for row in rows]),
                    stems=np.asarray(stems),
                    n_img=len(stems),
                    n_gt=np.asarray(
                        [len(row[3]) for row in rows],
                        dtype=np.int32,
                    ),
                )
                state = load_per_image(str(output_path))
                ap50, ap50_95 = pooled(state, np.arange(len(stems)))
                print(
                    f"{model_name:10s} {sequence} {candidate:8s} "
                    f"AP50={ap50:.6f} AP50-95={ap50_95:.6f} "
                    f"{time.time() - started:.1f}s",
                    flush=True,
                )

    metadata["pia_files"] = len(list(args.pia_dir.glob("*.png")))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
