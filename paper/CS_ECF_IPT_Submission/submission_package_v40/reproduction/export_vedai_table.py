"""Export the manuscript VEDAI table from canonical analysis evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ANALYSIS = (
    PACKAGE
    / "evidence"
    / "vedai"
    / "standard_matching"
    / "analysis.json"
)
OUTPUT = PACKAGE / "source_data" / "TableVedai_source.csv"


def main() -> None:
    payload = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    rows = []
    for run, record in payload["models"].items():
        comparisons = record["comparisons"]
        rows.append(
            {
                "scope": run,
                "visible_ap50": record["absolute_ap50"]["visible"],
                "cdf_ap50": record["absolute_ap50"]["always_cdf"],
                "no_shield_ap50": record["absolute_ap50"]["no_shield"],
                "csecf_ap50": record["absolute_ap50"]["csecf"],
                "csecf_minus_visible": comparisons[
                    "csecf_minus_visible"
                ]["equal_block_mean"],
                "csecf_minus_cdf": comparisons[
                    "csecf_minus_always_cdf"
                ]["equal_block_mean"],
                "csecf_minus_no_shield": comparisons[
                    "csecf_minus_no_shield"
                ]["equal_block_mean"],
            }
        )
    aggregate = payload["aggregate"]
    rows.append(
        {
            "scope": "equal_run_mean",
            "csecf_minus_visible": aggregate["csecf_minus_visible"][
                "equal_run_mean"
            ],
            "csecf_minus_cdf": aggregate["csecf_minus_always_cdf"][
                "equal_run_mean"
            ],
            "csecf_minus_no_shield": aggregate["csecf_minus_no_shield"][
                "equal_run_mean"
            ],
            "csecf_minus_visible_interval_low": aggregate[
                "csecf_minus_visible"
            ]["conditional_run_by_block_interval"][0],
            "csecf_minus_visible_interval_high": aggregate[
                "csecf_minus_visible"
            ]["conditional_run_by_block_interval"][1],
            "csecf_minus_no_shield_interval_low": aggregate[
                "csecf_minus_no_shield"
            ]["conditional_run_by_block_interval"][0],
            "csecf_minus_no_shield_interval_high": aggregate[
                "csecf_minus_no_shield"
            ]["conditional_run_by_block_interval"][1],
        }
    )
    columns = sorted({key for row in rows for key in row})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {OUTPUT}")


if __name__ == "__main__":
    main()
