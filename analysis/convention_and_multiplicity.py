"""Compare detection conventions and export BH-corrected headline contrasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_ap_reference import load, pooled


FUSED = [
    "mean", "activity_only", "tophat", "probe", "tophat_permuted",
    "probe_permuted", "tophat_wshuffle", "probe_wshuffle", "ensemble",
    "ifcnn", "densefuse", "csf", "piafusion", "spgfusion",
]
POOLED_VISIBLE = ["yolo11n_vis", "yolo26n_vis", "rtdetr_l_vis"]


def two_sided_p(draws: np.ndarray) -> float:
    positive = float(np.mean(draws >= 0))
    negative = float(np.mean(draws <= 0))
    return min(1.0, 2.0 * min(positive, negative))


def benjamini_hochberg(values: list[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = count - reverse_rank + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = running
    return adjusted


def summarise_scores(scores: dict[str, dict], alpha: float) -> dict:
    below = [name for name, row in scores.items()
             if row["lo50"] < 0 and row["q"] < alpha]
    above = [name for name, row in scores.items()
             if row["lo50"] > 0 and row["q"] < alpha]
    return {
        "below": len(below),
        "above": len(above),
        "unresolved": len(scores) - len(below) - len(above),
        "below_names": below,
        "above_names": above,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-image",
        default="_deposit/ssgf-llvip-fusion-detection-v1/data/per_image",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.per_image)
    rng = np.random.default_rng(7)
    output = {
        "protocol": {
            "bootstrap_draws": args.bootstrap,
            "families": {
                "pooled_gray": (
                    "42 single-channel contrasts over three visible-trained "
                    "evaluators"
                ),
                "pooled_colour": (
                    "42 colour-scored contrasts over three visible-trained "
                    "evaluators"
                ),
                "clean_split_gray": "14 single-channel clean-split contrasts",
                "clean_split_colour": "14 colour-scored clean-split contrasts",
            },
            "method": "Benjamini-Hochberg",
            "alpha": args.alpha,
            "random_seed": 7,
        },
        "pooled_visible": {},
        "clean_visible": {},
    }

    p_records = {"gray": [], "colour": []}
    pooled_rows = {}
    for evaluator in POOLED_VISIBLE:
        colour_ref = load(str(root / f"{evaluator}__vis.npz"))
        gray_ref = load(str(root / f"{evaluator}__vis_gray.npz"))
        n = colour_ref["n"]
        all_idx = np.arange(n)
        draws = [rng.choice(all_idx, size=n, replace=True)
                 for _ in range(args.bootstrap)]
        base_colour = np.asarray([pooled(colour_ref, draw)[0] for draw in draws])
        base_gray = np.asarray([pooled(gray_ref, draw)[0] for draw in draws])
        rows = {}
        for name in FUSED:
            stats = load(str(root / f"{evaluator}__{name}.npz"))
            values = np.asarray([pooled(stats, draw)[0] for draw in draws])
            for convention, base in (("colour", base_colour), ("gray", base_gray)):
                delta = values - base
                lo, hi = np.percentile(delta, [2.5, 97.5])
                rows[name] = rows.get(name, {})
                rows[name][convention] = {
                    "delta50": float(delta.mean()),
                    "lo50": float(lo),
                    "hi50": float(hi),
                    "p": two_sided_p(delta),
                }
            rows[name]["mAP50"] = float(pooled(stats, all_idx)[0])
            for convention in ("gray", "colour"):
                p_records[convention].append(
                    (evaluator, name, rows[name][convention]["p"])
                )
        pooled_rows[evaluator] = rows

    q_lookup = {}
    for convention, records in p_records.items():
        q_values = benjamini_hochberg([record[2] for record in records])
        q_lookup[convention] = {
            (evaluator, name): float(q)
            for (evaluator, name, _), q in zip(records, q_values)
        }
    for evaluator in POOLED_VISIBLE:
        rows = pooled_rows[evaluator]
        for name in FUSED:
            for convention in ("gray", "colour"):
                rows[name][convention]["q"] = q_lookup[convention][
                    (evaluator, name)
                ]
        output["pooled_visible"][evaluator] = {
            "contours": rows,
            "q_count": {
                convention: int(sum(
                    q_lookup[convention][(evaluator, name)] < args.alpha
                    for name in FUSED
                ))
                for convention in ("gray", "colour")
            },
        }

    clean = "vis_split"
    colour_ref = load(str(root / f"{clean}__vis.npz"))
    gray_ref = load(str(root / f"{clean}__vis_gray.npz"))
    n = colour_ref["n"]
    all_idx = np.arange(n)
    draws = [rng.choice(all_idx, size=n, replace=True)
             for _ in range(args.bootstrap)]
    base_colour = np.asarray([pooled(colour_ref, draw)[0] for draw in draws])
    base_gray = np.asarray([pooled(gray_ref, draw)[0] for draw in draws])
    rows = {}
    clean_p_values = {"gray": [], "colour": []}
    for name in FUSED:
        stats = load(str(root / f"{clean}__{name}.npz"))
        values = np.asarray([pooled(stats, draw)[0] for draw in draws])
        row = {"mAP50": float(pooled(stats, all_idx)[0])}
        for convention, base in (("colour", base_colour), ("gray", base_gray)):
            delta = values - base
            lo, hi = np.percentile(delta, [2.5, 97.5])
            row[convention] = {
                "delta50": float(delta.mean()),
                "lo50": float(lo),
                "hi50": float(hi),
                "p": two_sided_p(delta),
            }
            clean_p_values[convention].append(row[convention]["p"])
        rows[name] = row
    clean_q = {
        convention: benjamini_hochberg(values)
        for convention, values in clean_p_values.items()
    }
    for name, gray_q, colour_q in zip(
        FUSED, clean_q["gray"], clean_q["colour"]
    ):
        rows[name]["gray"]["q"] = float(gray_q)
        rows[name]["colour"]["q"] = float(colour_q)
    output["clean_visible"] = {
        "contours": rows,
        "colour": summarise_scores(
            {name: rows[name]["colour"] for name in FUSED}, args.alpha
        ),
        "gray": summarise_scores(
            {name: rows[name]["gray"] for name in FUSED}, args.alpha
        ),
    }

    for evaluator in POOLED_VISIBLE:
        rows = output["pooled_visible"][evaluator]["contours"]
        output["pooled_visible"][evaluator]["colour"] = summarise_scores(
            {name: rows[name]["colour"] for name in FUSED}, args.alpha
        )
        output["pooled_visible"][evaluator]["gray"] = summarise_scores(
            {name: rows[name]["gray"] for name in FUSED}, args.alpha
        )
        output["pooled_visible"][evaluator]["details"] = rows
        del output["pooled_visible"][evaluator]["contours"]

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "pooled": {
            evaluator: {
                convention: {
                    key: value for key, value in
                    output["pooled_visible"][evaluator][convention].items()
                    if key in ("below", "above", "unresolved")
                }
                for convention in ("gray", "colour")
            }
            for evaluator in POOLED_VISIBLE
        },
        "clean_visible": {
            convention: {
                key: value for key, value in
                output["clean_visible"][convention].items()
                if key in ("below", "above", "unresolved")
            }
            for convention in ("gray", "colour")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
