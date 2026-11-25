"""Train replication seeds for the two M3FD six-class evaluators."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
COMPAT = next(
    path
    for path in (
        SCRIPT_PATH.parent / "torchvision_cuda_compat.py",
        *(root / "scripts" / "torchvision_cuda_compat.py"
          for root in SCRIPT_PATH.parents
          if root.name != "scripts"),
    )
    if path.is_file()
)
sys.path.insert(0, str(COMPAT.parent))
import torchvision_cuda_compat  # noqa: F401,E402
from ultralytics import YOLO


DEFAULT_ROOT = Path(r"D:\Code\_ssgf_external\m3fd_erf_pilot")
MODELS = {
    "yolo26n_m3fd_6cls": Path(r"D:\Code\_ssgf_external\r2_clean\weights\yolo26n.pt"),
    "yolo8n_m3fd_6cls": Path(r"D:\Code\_ssgf_external\r2_clean\weights\yolov8n.pt"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODELS),
        help="Restrict training to one or more baseline model families.",
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    models = args.model or list(MODELS)
    data = args.root / "det" / "data.yaml"
    manifest_path = args.root / "runs" / "multiseed_manifest.json"
    existing_runs = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_runs = {
            (run["model"], int(run["seed"])): run
            for run in existing.get("runs", [])
        }
    for model_name in models:
        initialization = MODELS[model_name]
        if not initialization.is_file():
            raise FileNotFoundError(initialization)
        for seed in seeds:
            run_name = f"{model_name}_s{seed}"
            started = time.time()
            model = YOLO(str(initialization))
            result = model.train(
                data=str(data),
                epochs=args.epochs,
                imgsz=640,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                seed=seed,
                deterministic=True,
                pretrained=True,
                amp=False,
                project=str(args.root / "runs"),
                name=run_name,
                exist_ok=False,
                val=True,
            )
            save_dir = Path(result.save_dir)
            run = {
                "model": model_name,
                "seed": seed,
                "run_name": run_name,
                "save_dir": str(save_dir),
                "last_weights": str(save_dir / "weights" / "last.pt"),
                "best_weights": str(save_dir / "weights" / "best.pt"),
                "seconds": time.time() - started,
            }
            existing_runs[(model_name, seed)] = run
    results = [
        existing_runs[key] for key in sorted(existing_runs)
    ]
    manifest = {
        "data": str(data),
        "epochs": args.epochs,
        "device": args.device,
        "batch": args.batch,
        "workers": args.workers,
        "initialization": {
            name: str(path) for name, path in MODELS.items()
        },
        "runs": results,
    }
    out = manifest_path
    out.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
