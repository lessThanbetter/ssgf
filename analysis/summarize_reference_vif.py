"""Combine reported and reference VIF scores into one benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_scores(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["variant"]: row for row in payload["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="DATASET=reported.json,reference.json",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for spec in args.spec:
        dataset, paths = spec.split("=", 1)
        reported_path, reference_path = map(Path, paths.split(",", 1))
        reported = read_scores(reported_path)
        reference = read_scores(reference_path)
        for variant in sorted(set(reported) & set(reference)):
            left, right = reported[variant], reference[variant]
            if left["n"] != right["n"]:
                raise SystemExit(
                    f"{dataset}/{variant}: n mismatch {left['n']} != {right['n']}"
                )
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "n": left["n"],
                "reported": left["reported"],
                "reference": right["reference"],
                "difference": right["reference"] - left["reported"],
            })

    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        for field in ("reported", "reference"):
            order = sorted(subset, key=lambda row: -row[field])
            for rank, row in enumerate(order, start=1):
                row[f"{field}_rank"] = rank

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    destination.write_text(
        json.dumps({
            "definitions": {
                "reported": "width 2**(s-1), no decimation, noise 2.0 in [0,1]",
                "reference": "canonical widths and decimation, noise 2/255**2",
            },
            "rows": rows,
        }, indent=2),
        encoding="utf-8",
    )

    print("| Dataset | Method | reported | reference | rank change |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['dataset']} | {row['variant']} | "
            f"{row['reported']:.4f} | {row['reference']:.4f} | "
            f"{row['reported_rank']} -> {row['reference_rank']} |"
        )


if __name__ == "__main__":
    main()
