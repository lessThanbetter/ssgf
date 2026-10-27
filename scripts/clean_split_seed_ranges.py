"""Summarise the clean-split detector runs without resampling images.

The primary clean-split comparison uses four independent training runs.  This
script records the mean contrast over runs and the observed range across runs;
the image-level paired intervals remain in the aggregate contrast files.  Keeping
the two uncertainty sources separate avoids presenting a seed range as a 95 %
confidence interval.
"""
import json
import os

import numpy as np

from _local_ap import load, pooled


BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_restore_perimg")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "_clean_split_seed_ranges.json")
FUSED = [
    "mean", "activity_only", "tophat", "probe",
    "tophat_permuted", "probe_permuted",
    "tophat_wshuffle", "probe_wshuffle", "ensemble",
    "ifcnn", "densefuse", "csf", "piafusion", "spgfusion",
]


def summarise(family):
    tags = [family, family + "_s1", family + "_s2", family + "_s3"]
    values = {}
    for tag in tags:
        reference = load(os.path.join(BASE, tag + "__vis.npz"))
        idx = np.arange(reference["n"])
        values[tag] = {"vis": pooled(reference, idx)[0]}
        for name in FUSED:
            fused = load(os.path.join(BASE, tag + "__" + name + ".npz"))
            values[tag][name] = pooled(fused, idx)[0]

    out = {
        "runs": tags,
        "visible_source_mAP50": float(np.mean([values[t]["vis"] for t in tags])),
        "inputs": {},
    }
    for name in FUSED:
        contrast = np.asarray([
            values[tag][name] - values[tag]["vis"] for tag in tags
        ])
        out["inputs"][name] = {
            "mean_mAP50": float(np.mean([values[tag][name] for tag in tags])),
            "mean_contrast": float(contrast.mean()),
            "min_contrast": float(contrast.min()),
            "max_contrast": float(contrast.max()),
            "per_run_contrast": [float(x) for x in contrast],
        }
    return out


def main():
    payload = {family: summarise(family)
               for family in ("vis_split", "agn_split")}
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(OUT)


if __name__ == "__main__":
    main()
