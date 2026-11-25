"""Attach annotation counts and contribution shares to M3FD class effects."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SSGF_DATA_ROOT", str(SCRIPT_ROOT)))
sys.path.insert(0, str(DATA_ROOT / "scripts"))

import analyze_csecf_vs_cdf as csecf  # noqa: E402


CLASS_NAMES = {
    0: "people",
    1: "car",
    2: "bus",
    3: "motorcycle",
    4: "lamp",
    5: "truck",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = next(
        spec for spec in csecf._candidate_paths()
        if spec.dataset == "m3fd"
    )
    states = csecf._load_checkpoint(spec)
    classes, counts = np.unique(states["vis"]["gt_all"], return_counts=True)
    annotation_counts = {
        CLASS_NAMES[int(class_id)]: int(count)
        for class_id, count in zip(classes, counts)
    }
    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    intervals = json.loads(args.intervals.read_text(encoding="utf-8"))
    output = {
        "annotations": annotation_counts,
        "total_annotations": int(sum(annotation_counts.values())),
        "models": {},
    }
    for model, class_rows in diagnostics["m3fd"].items():
        gains = np.asarray(list(
            class_rows["gain_vs_visible"]["confidence_shield"].values()
        ))
        total_positive_gain = float(gains[gains > 0].sum())
        output["models"][model] = {}
        for class_id, class_name in diagnostics["m3fd"][model]["classes"].items():
            gain = class_rows["gain_vs_visible"]["confidence_shield"][class_id]
            output["models"][model][class_name] = {
                "annotations": annotation_counts[class_name],
                "visible_ap50": class_rows["ap50"]["visible"][class_id],
                "csecf_ap50": class_rows["ap50"]["confidence_shield"][class_id],
                "gain": gain,
                "image_ci95": intervals["m3fd"][model][class_id]["image_ci95"],
                "share_of_positive_class_mean_gain": (
                    float(gain / total_positive_gain)
                    if gain > 0 and total_positive_gain > 0
                    else 0.0
                ),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
