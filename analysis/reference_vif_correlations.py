"""Correlate reported and reference VIF with downstream detection contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import spearmanr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vif", required=True)
    parser.add_argument("--contrasts", required=True,
                        help="aggregate directory with contrasts_*_colour.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.vif).read_text(encoding="utf-8"))
    vif = {row["variant"]: row for row in payload["rows"]}
    rows = []
    for path in sorted(Path(args.contrasts).glob("contrasts_*_colour.json")):
        detection = json.loads(path.read_text(encoding="utf-8"))
        names = [name for name in vif if name in detection]
        if len(names) < 3:
            continue
        row = {"evaluator": path.stem.removeprefix("contrasts_").removesuffix(
            "_colour"), "n": len(names)}
        for field in ("reported", "reference"):
            result = spearmanr(
                [vif[name][field] for name in names],
                [detection[name]["delta50"] for name in names],
            )
            row[field] = {
                "rho": float(result.statistic),
                "p": float(result.pvalue),
            }
        rows.append(row)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['evaluator']:<18} n={row['n']:2d} "
            f"reported={row['reported']['rho']:+.3f} "
            f"reference={row['reference']['rho']:+.3f}"
        )


if __name__ == "__main__":
    main()
