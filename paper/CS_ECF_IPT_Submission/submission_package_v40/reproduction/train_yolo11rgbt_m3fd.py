"""Train a recent single-stage RGB-T detector on the fixed M3FD split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--resume-weights", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="yolo11n_rgbt_earlyfusion")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    import torch
    from ultralytics import YOLO

    if args.resume_weights:
        model = YOLO(str(args.resume_weights))
    else:
        model = YOLO(str(args.model))
        model.load(str(args.initial_weights))
        source_checkpoint = torch.load(
            args.initial_weights,
            map_location="cpu",
            weights_only=False,
        )
        source_model = source_checkpoint.get("ema") or source_checkpoint["model"]
        source_conv = source_model.model[0].conv
        target_conv = model.model.model[2].conv
        if source_conv.weight.shape[1] != 3 or target_conv.weight.shape[1] != 4:
            raise ValueError("Expected a three-channel source and four-channel target stem")
        with torch.no_grad():
            target_conv.weight[:, :3].copy_(source_conv.weight)
            target_conv.weight[:, 3:].copy_(
                source_conv.weight.mean(dim=1, keepdim=True)
            )
    augment_values = (
        {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "fliplr": 0.5,
            "scale": 0.5,
            "translate": 0.1,
        }
        if args.augment
        else {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "fliplr": 0.0,
            "scale": 0.0,
            "translate": 0.0,
        }
    )
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        close_mosaic=10,
        seed=0,
        deterministic=True,
        amp=args.amp,
        mosaic=args.mosaic,
        auto_augment=None,
        erasing=0.0,
        **augment_values,
        brightness=0.0,
        mixup=0.0,
        copy_paste=0.0,
        single_cls=False,
        use_simotm="RGBT",
        channels=4,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        plots=False,
    )

    run_dir = args.project / args.name
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)

    model = YOLO(str(best))
    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=640,
        conf=0.001,
        iou=0.7,
        max_det=300,
        single_cls=False,
        use_simotm="RGBT",
        channels=4,
        project=str(args.project),
        name=f"{args.name}_test",
        exist_ok=True,
        plots=False,
        save_json=True,
        verbose=True,
    )

    payload = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "method": "YOLOv11-RGBT early fusion",
        "framework_commit": "9cc2e208a3d3e8452e15b5ef47e1c536766aa1f7",
        "model_yaml": str(args.model.resolve()),
        "initial_weights": str(args.initial_weights.resolve()),
        "initial_weights_sha256": sha256(args.initial_weights),
        "resume_weights": (
            str(args.resume_weights.resolve()) if args.resume_weights else None
        ),
        "resume_weights_sha256": (
            sha256(args.resume_weights) if args.resume_weights else None
        ),
        "best_weights": str(best.resolve()),
        "best_weights_sha256": sha256(best),
        "data": str(args.data.resolve()),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "input_size": 640,
            "optimizer": args.optimizer,
            "seed": 0,
            "single_class": False,
            "input_channels": 4,
            "amp": args.amp,
            "mosaic": args.mosaic,
            "augmentation": (
                "mosaic, horizontal flip, scale, translation, and HSV"
                if args.mosaic > 0 and args.augment
                else (
                    "horizontal flip, scale, translation, and HSV"
                    if args.augment
                    else "resize only"
                )
            ),
            "initialization": "3-channel COCO YOLO11n weights with the input stem adapted to four channels",
        },
        "test_metrics": {
            "ap50_95": float(metrics.box.map),
            "ap50": float(metrics.box.map50),
            "ap75": float(metrics.box.map75),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
