"""Record the training recipe and validation endpoint of one clean-split run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import ultralytics
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = Path(args.run)
    results_path = run / "results.csv"
    args_path = run / "args.yaml"
    best = run / "weights" / "best.pt"
    last = run / "weights" / "last.pt"
    for path in (results_path, args_path, best, last):
        if not path.is_file():
            raise SystemExit(f"missing training artifact: {path}")

    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("results.csv is empty")

    def find_key(prefix: str) -> str:
        matches = [key for key in rows[0] if key.strip() == prefix]
        if len(matches) != 1:
            raise SystemExit(f"expected one results column {prefix!r}, got {matches}")
        return matches[0]

    map50_key = find_key("metrics/mAP50(B)")
    map5095_key = find_key("metrics/mAP50-95(B)")
    best_row = max(
        rows,
        key=lambda row: 0.1 * float(row[map50_key])
        + 0.9 * float(row[map5095_key]),
    )
    peak_map50_row = max(rows, key=lambda row: float(row[map50_key]))
    training_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    payload = {
        "name": args.name,
        "run_directory": str(run),
        "recipe": {
            key: training_args.get(key)
            for key in (
                "model",
                "data",
                "epochs",
                "imgsz",
                "batch",
                "seed",
                "device",
                "workers",
                "optimizer",
                "pretrained",
                "deterministic",
            )
        },
        "epochs_completed": int(float(rows[-1]["epoch"])),
        "training_seconds": float(rows[-1]["time"]),
        "software": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "best_validation_epoch": int(float(best_row["epoch"])),
        "best_validation_fitness": round(
            0.1 * float(best_row[map50_key])
            + 0.9 * float(best_row[map5095_key]),
            6,
        ),
        "best_validation_mAP50": float(best_row[map50_key]),
        "best_validation_mAP50_95": float(best_row[map5095_key]),
        "peak_validation_mAP50": float(peak_map50_row[map50_key]),
        "peak_validation_mAP50_epoch": int(float(peak_map50_row["epoch"])),
        "final_validation_mAP50": float(rows[-1][map50_key]),
        "final_validation_mAP50_95": float(rows[-1][map5095_key]),
        "checkpoints": {
            "best.pt": {
                "bytes": best.stat().st_size,
                "sha256": sha256(best),
            },
            "last.pt": {
                "bytes": last.stat().st_size,
                "sha256": sha256(last),
            },
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
