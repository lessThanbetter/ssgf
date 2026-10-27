"""Test detection contrasts on the best- and worst-aligned source quartiles.

Selection uses only the registered source pair.  It never uses labels, detector
scores, or fusion fidelity, so the resulting subset is an independent sensitivity
check rather than a post-hoc selection of favourable images.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_ap_reference import load, pooled


def read_alignment(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = []
    for row in rows:
        if row["status"] != "ok":
            continue
        row["residual_px"] = float(row["residual_px"])
        row["ecc"] = float(row["ecc"])
        valid.append(row)
    return valid


def select_groups(rows: list[dict], fraction: float,
                  min_ecc: float) -> dict[str, list[str]]:
    usable = [row for row in rows if row["ecc"] >= min_ecc]
    if len(usable) < 8:
        usable = rows
    ordered = sorted(usable, key=lambda row: (row["residual_px"], -row["ecc"]))
    count = max(4, round(len(ordered) * fraction))
    return {
        "best": [row["stem"] for row in ordered[:count]],
        "worst": [row["stem"] for row in ordered[-count:]],
        "all_valid": [row["stem"] for row in ordered],
    }


def group_context(path: Path | None, groups: dict[str, list[str]]) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["stem"]: row for row in csv.DictReader(handle)}
    context = {}
    for group, stems in groups.items():
        selected = [rows[stem] for stem in stems if stem in rows]
        if not selected:
            continue
        context[group] = {
            "n": len(selected),
            "visible_rms_contrast_median": float(np.median(
                [float(row["visible_rms_contrast"]) for row in selected]
            )),
            "visible_mean_luminance_median": float(np.median(
                [float(row["visible_mean_luminance"]) for row in selected]
            )),
            "pedestrians_per_image_mean": float(np.mean(
                [float(row["n_pedestrians"]) for row in selected]
            )),
            "median_box_area_px_median": float(np.median(
                [float(row["median_box_area_px"]) for row in selected]
            )),
        }
    return context


def evaluate_group(stats: dict, stems: list[str], baseline: str,
                   inputs: list[str], bootstrap: int, seed: int):
    stem_to_index = {stem: i for i, stem in enumerate(stats[baseline]["stems"])}
    selected = np.asarray(
        [stem_to_index[stem] for stem in stems if stem in stem_to_index],
        dtype=np.int64,
    )
    if len(selected) == 0:
        raise RuntimeError("none of the selected stems are present in the NPZ files")
    rng = np.random.default_rng(seed)
    masks = [
        np.unique(rng.choice(selected, size=len(selected), replace=True))
        for _ in range(bootstrap)
    ]
    base = np.asarray([pooled(stats[baseline], mask)[0] for mask in masks])
    result = {
        "n": int(len(selected)),
        "stems": [stats[baseline]["stems"][i] for i in selected],
        "scores": {},
    }
    for name in inputs:
        if name not in stats:
            continue
        point = pooled(stats[name], selected)[0]
        score = np.asarray([pooled(stats[name], mask)[0] for mask in masks])
        result["scores"][name] = {"mAP50": float(point)}
        if name != baseline:
            delta = score - base
            lo, hi = np.percentile(delta, [2.5, 97.5])
            result["scores"][name].update({
                "delta50": float(delta.mean()),
                "lo50": float(lo),
                "hi50": float(hi),
                "resolved": bool(lo > 0 or hi < 0),
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--per-image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--regime-csv", default="")
    parser.add_argument("--evaluators", nargs="+",
                        default=["yolo11n_vis", "agnostic"])
    parser.add_argument("--inputs", nargs="+",
                        default=["mean", "tophat", "probe", "ensemble",
                                 "ifcnn", "densefuse", "piafusion",
                                 "spgfusion"])
    parser.add_argument("--baseline", default="vis")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--min-ecc", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    rows = read_alignment(Path(args.alignment))
    groups = select_groups(rows, args.fraction, args.min_ecc)
    regime_path = Path(args.regime_csv) if args.regime_csv else None
    per_image = Path(args.per_image)
    output = {}
    for evaluator_index, evaluator in enumerate(args.evaluators):
        stats = {}
        for path in sorted(per_image.glob(f"{evaluator}__*.npz")):
            name = path.stem[len(evaluator) + 2:]
            stats[name] = load(str(path))
        if args.baseline not in stats:
            raise SystemExit(f"{args.baseline} is missing for {evaluator}")
        output[evaluator] = {}
        for group_index, (group, stems) in enumerate(groups.items()):
            print(f"[alignment-test] {evaluator} {group}: {len(stems)} images",
                  flush=True)
            output[evaluator][group] = evaluate_group(
                stats,
                stems,
                args.baseline,
                args.inputs,
                args.bootstrap,
                seed=2026 + evaluator_index * 10 + group_index,
            )

    valid = [row for row in rows if np.isfinite(row["residual_px"])]
    payload = {
        "selection": {
            "method": "ECC translation residual on normalized Sobel edges",
            "fraction": args.fraction,
            "min_ecc": args.min_ecc,
            "source_rows": len(rows),
            "residual_median_px": float(np.median(
                [row["residual_px"] for row in valid]
            )),
            "residual_best_cutoff_px": float(max(
                row["residual_px"]
                for row in rows
                if row["stem"] in groups["best"]
            )),
            "group_context": group_context(regime_path, groups),
            "bootstrap_draws": args.bootstrap,
            "random_seed": 2026,
        },
        "evaluators": output,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[alignment-test] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
