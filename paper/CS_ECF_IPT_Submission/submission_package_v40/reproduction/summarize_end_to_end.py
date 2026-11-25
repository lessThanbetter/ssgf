"""Summarize the recorded end-to-end latency benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE = SCRIPT_PATH.parents[1]
STYLES = ("visible", "budgeted", "full")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PACKAGE
            / "evidence"
            / "prospective_pia_confirmation"
            / "end_to_end_latency.json"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=(
            PACKAGE
            / "source_data"
            / "TableEndToEndLatency_source.csv"
        ),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = payload["results"]
    rows = []
    for key, value in results.items():
        model, style = key.split("/", 1)
        rows.append(
            {
                "model": model,
                "style": style,
                "median_total_ms": value["median_total_ms"],
                "p95_total_ms": value["p95_total_ms"],
                "median_fusion_ms": value["median_fusion_ms"],
                "median_detector_ms": value["median_detector_ms"],
            }
        )
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for style in STYLES:
        values = [
            row["median_total_ms"]
            for row in rows
            if row["style"] == style
        ]
        p95 = [
            row["p95_total_ms"]
            for row in rows
            if row["style"] == style
        ]
        summary[style] = {
            "equal_checkpoint_median_total_ms": float(np.median(values)),
            "checkpoint_median_range_ms": [float(min(values)), float(max(values))],
            "equal_checkpoint_median_p95_ms": float(np.median(p95)),
        }
    payload["summary"] = summary
    args.input.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
