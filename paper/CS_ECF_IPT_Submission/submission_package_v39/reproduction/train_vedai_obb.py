"""Train fixed VEDAI YOLO11n-OBB runs using official folds."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torchvision_cuda_compat  # noqa: F401,E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\vedai_fold01\data.yaml"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(r"D:\Code\_server_handoff_v2\ssgf\yolo11n-obb.pt"),
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(r"D:\Code\_server_handoff_v2\ssgf\results\vedai_20261029"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = YOLO(str(args.weights))
    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=0,
        seed=args.seed,
        deterministic=True,
        amp=True,
        cache=False,
        patience=args.epochs,
        project=str(args.project),
        name=args.run_name,
        exist_ok=False,
        close_mosaic=5,
        verbose=True,
    )
    save_dir = Path(result.save_dir)
    manifest = {
        "run": args.run_name,
        "seed": args.seed,
        "data": str(args.data),
        "weights_initialization": str(args.weights),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "save_dir": str(save_dir),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "checkpoint_used_for_test": "last.pt",
    }
    (args.project / f"{args.run_name}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
