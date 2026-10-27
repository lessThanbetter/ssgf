"""Train one clean-split YOLO or RT-DETR evaluator for reviewer item R2-2."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR, YOLO

import torchvision_cuda_compat  # noqa: F401

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("yolo", "rtdetr"), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cls = RTDETR if args.kind == "rtdetr" else YOLO
    model = cls(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=0,
        workers=args.workers,
        amp=args.amp,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"training finished without {best}")
    print(best)


if __name__ == "__main__":
    main()
