"""Evaluate CS-ECF on best- and worst-aligned LLVIP quartiles."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SSGF_DATA_ROOT", str(SCRIPT_ROOT)))
sys.path.insert(0, str(DATA_ROOT / "scripts"))

import analyze_csecf_vs_cdf as csecf  # noqa: E402
from local_ap_reference import pooled as local_pooled  # noqa: E402


def _ap(stats: dict, indices: np.ndarray) -> float:
    return float(local_pooled(stats, np.asarray(indices, dtype=np.int64))[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--alignment-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--min-ecc", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    with args.alignment_csv.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["status"] == "ok"
        ]
    for row in rows:
        row["residual_px"] = float(row["residual_px"])
        row["ecc"] = float(row["ecc"])
    usable = [row for row in rows if row["ecc"] >= args.min_ecc]
    if len(usable) < 8:
        usable = rows
    ordered = sorted(usable, key=lambda row: (row["residual_px"], -row["ecc"]))
    count = max(4, round(len(ordered) * args.fraction))
    subsets = {
        "best_aligned": [row["stem"] for row in ordered[:count]],
        "worst_aligned": [row["stem"] for row in ordered[-count:]],
    }

    llvip_specs = [
        spec for spec in csecf._candidate_paths()
        if spec.dataset == "llvip"
    ]
    output = {
        "selection": {
            "method": "ECC translation residual on normalized Sobel edges",
            "fraction": args.fraction,
            "min_ecc": args.min_ecc,
            "valid_pairs": len(rows),
            "subset_size": count,
            "median_residual_px": float(np.median(
                [row["residual_px"] for row in usable]
            )),
            "best_residual_cutoff_px": max(
                row["residual_px"]
                for row in usable
                if row["stem"] in subsets["best_aligned"]
            ),
            "worst_residual_min_px": min(
                row["residual_px"]
                for row in usable
                if row["stem"] in subsets["worst_aligned"]
            ),
        },
        "checkpoints": {},
        "summary": {},
    }

    for group_index, (group, stems) in enumerate(subsets.items()):
        gains = []
        output["summary"][group] = {}
        for model_index, spec in enumerate(llvip_specs):
            states = csecf._load_checkpoint(spec)
            choices, _ = csecf._decisions(states)
            selected_stats = csecf._policy_stats(states, choices)
            stem_to_index = {
                str(stem): index
                for index, stem in enumerate(states["vis"]["stems"])
            }
            indices = np.asarray([
                stem_to_index[stem] for stem in stems
                if stem in stem_to_index
            ])
            gain = (
                _ap(selected_stats, indices)
                - _ap(states["vis"], indices)
            )
            gains.append(gain)
            rng = np.random.default_rng(
                9100 + group_index * 100 + model_index
            )
            draws = []
            for _ in range(args.bootstrap):
                sampled = rng.choice(indices, size=len(indices), replace=True)
                draws.append(
                    _ap(selected_stats, sampled)
                    - _ap(states["vis"], sampled)
                )
            output["checkpoints"][f"{group}/{spec.name}"] = {
                "n_images": len(indices),
                "gain": float(gain),
                "image_bootstrap_ci95": [
                    float(value)
                    for value in np.percentile(draws, [2.5, 97.5])
                ],
            }
        output["summary"][group] = {
            "mean_gain": float(np.mean(gains)),
            "min_gain": float(np.min(gains)),
            "max_gain": float(np.max(gains)),
            "positive_checkpoints": int(np.sum(np.asarray(gains) > 0)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
