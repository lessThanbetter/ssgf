"""Benjamini-Hochberg correction over the overlapping test-set strata."""

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


def two_sided_p(draws: np.ndarray) -> float:
    return min(1.0, 2.0 * min(float(np.mean(draws >= 0)),
                              float(np.mean(draws <= 0))))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    parser.add_argument(
        "--per-image",
        default="_deposit/ssgf-llvip-fusion-detection-v1/data/per_image",
    )
    parser.add_argument(
        "--stats",
        default="_deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_stats.json",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    per_image = Path(args.per_image)
    reference = load(str(per_image / f"{args.detector}__vis.npz"))
    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    stems = reference["stems"]
    rms = np.asarray([stats[stem]["rms_contrast"] for stem in stems])
    count = np.asarray([stats[stem]["n_gt"] for stem in stems])
    area = np.asarray([stats[stem]["median_area"] for stem in stems])
    rms_low, rms_high = np.percentile(rms, [33.3, 66.7])
    area_median = float(np.median(area[area > 0]))
    regimes = {
        "low contrast": rms <= rms_low,
        "high contrast": rms >= rms_high,
        "one pedestrian": count == 1,
        "three or more": count >= 3,
        "small targets": (area > 0) & (area <= area_median),
        "large targets": area > area_median,
    }

    rng = np.random.default_rng(7)
    tests = []
    for regime, mask in regimes.items():
        indices = np.nonzero(mask)[0]
        draws = [rng.choice(indices, size=len(indices), replace=True)
                 for _ in range(args.bootstrap)]
        base = np.asarray([pooled(reference, draw)[0] for draw in draws])
        for name in FUSED:
            current = load(str(per_image / f"{args.detector}__{name}.npz"))
            values = np.asarray([pooled(current, draw)[0] for draw in draws])
            delta = values - base
            lo, hi = np.percentile(delta, [2.5, 97.5])
            tests.append({
                "regime": regime,
                "input": name,
                "n": len(indices),
                "delta": float(delta.mean()),
                "lo": float(lo),
                "hi": float(hi),
                "p": two_sided_p(delta),
                "mAP50": float(pooled(current, indices)[0]),
            })

    q_values = benjamini_hochberg([row["p"] for row in tests])
    for row, q in zip(tests, q_values):
        row["q"] = float(q)

    summary = {}
    for regime in regimes:
        rows = [row for row in tests if row["regime"] == regime]
        below = [row["input"] for row in rows
                 if row["q"] < args.alpha and row["lo"] < 0]
        above = [row["input"] for row in rows
                 if row["q"] < args.alpha and row["lo"] > 0]
        summary[regime] = {
            "n": rows[0]["n"],
            "below": len(below),
            "above": len(above),
            "unresolved": len(rows) - len(below) - len(above),
            "below_names": below,
            "above_names": above,
        }

    payload = {
        "detector": args.detector,
        "protocol": {
            "bootstrap_draws": args.bootstrap,
            "family": f"{len(FUSED)} inputs x {len(regimes)} overlapping strata = {len(tests)} tests",
            "method": "Benjamini-Hochberg",
            "alpha": args.alpha,
            "random_seed": 7,
        },
        "thresholds": {
            "visible_rms_contrast": [float(rms_low), float(rms_high)],
            "median_box_area_px": area_median,
        },
        "summary": summary,
        "tests": tests,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "detector": args.detector,
        "thresholds": payload["thresholds"],
        "summary": {
            name: {key: value for key, value in row.items()
                   if key in ("n", "below", "above", "unresolved")}
            for name, row in summary.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
