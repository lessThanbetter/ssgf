"""Compute fidelity-to-detection rank associations for all clean evaluators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import spearmanr


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vif", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    vif_rows = read_json(Path(args.vif))["rows"]
    summary = read_json(Path(args.summary))
    evaluator_keys = [
        "yolo11n_visible_clean",
        "yolo26n_visible_clean",
        "rtdetr_l_visible_clean",
        "yolo11n_modality_agnostic_clean",
    ]
    rows = []
    for evaluator in evaluator_keys:
        inputs = summary["evaluators"][evaluator]["inputs"]
        contrasts = [inputs[row["variant"]]["mean_contrast"] for row in vif_rows]
        rendered = {
            "evaluator": evaluator,
            "n": len(vif_rows),
            "training_runs": summary["evaluators"][evaluator]["training_runs"],
        }
        for construction in ("reported", "reference"):
            scores = [row[construction] for row in vif_rows]
            association = spearmanr(scores, contrasts)
            rendered[construction] = {
                "rho": float(association.statistic),
                "p": float(association.pvalue),
            }
        rows.append(rendered)

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
