"""Train independent DroneVehicle OBB runs on disjoint frame groups."""

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
    parser.add_argument("--run", choices=("run1", "run2"), required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\Code\_ssgf_external\dronevehicle_val300"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(r"D:\Code\_server_handoff_v2\ssgf\yolo11n-obb.pt"),
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(
            r"D:\Code\_server_handoff_v2\ssgf\results\prospective_20261029"
            r"\dronevehicle"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    seed = 0 if args.run == "run1" else 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data = args.data_root / "det_obb" / f"data_{args.run}.yaml"
    model = YOLO(str(args.weights))
    result = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=0,
        seed=seed,
        deterministic=True,
        amp=True,
        cache=False,
        patience=10,
        project=str(args.project),
        name=args.run,
        exist_ok=False,
        close_mosaic=5,
        verbose=True,
    )
    save_dir = Path(result.save_dir)
    manifest = {
        "run": args.run,
        "seed": seed,
        "data": str(data),
        "weights_initialization": str(args.weights),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "save_dir": str(save_dir),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "best_weights": str(save_dir / "weights" / "best.pt"),
    }
    (args.project / f"{args.run}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
