"""Export the main-text joint-baseline comparison from canonical evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    joint = json.loads(args.joint.read_text(encoding="utf-8"))
    statistics = json.loads(args.statistics.read_text(encoding="utf-8"))
    checkpoints = statistics["checkpoints"]
    rows = []
    for key, label in (
        ("m3fd/yolo26_m3fd6", "YOLO26n + visible"),
        ("m3fd/yolo8n_m3fd6", "YOLOv8n + visible"),
    ):
        checkpoint = checkpoints[key]
        rows.append(
            {
                "system": label,
                "setting": "visible-only checkpoint",
                "ap50": round(checkpoint["point"]["visible"], 6),
            }
        )
        rows.append(
            {
                "system": label.replace("+ visible", "+ CS-ECF"),
                "setting": "same checkpoint with confidence shield",
                "ap50": round(checkpoint["point"]["csecf"], 6),
            }
        )
    rows.append(
        {
            "system": "YOLOv11-RGBT early fusion",
            "setting": "joint four-channel fusion and detection",
            "ap50": round(joint["metric"]["ap50"], 6),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("system", "setting", "ap50"))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
