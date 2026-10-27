"""Per-run point estimates for the RT-DETR-l repetition, to match how Table S14 reports."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_ap import load, pooled  # noqa: E402

PERIMG = r"D:\CSF-main\ssgf\_restore_perimg"
FUSED = ["mean", "activity_only", "tophat", "probe", "tophat_permuted", "probe_permuted",
         "tophat_wshuffle", "probe_wshuffle", "ensemble", "ifcnn", "densefuse", "csf",
         "piafusion", "spgfusion", "vis_gray", "ir"]
TAGS = ["rtdetr_l_vis", "rtdetr_l_vis_s1", "rtdetr_l_vis_s2"]

idx = None
out = {}
for tag in TAGS:
    ref = load(os.path.join(PERIMG, "%s__vis.npz" % tag))
    if idx is None:
        idx = np.arange(len(ref["stems"]))
    base = pooled(ref, idx)[0]
    out[tag] = {"vis": base}
    for name in FUSED:
        p = os.path.join(PERIMG, "%s__%s.npz" % (tag, name))
        if not os.path.exists(p):
            continue
        out[tag][name] = pooled(load(p), idx)[0] - base

print("%-24s %10s %10s %10s %10s" % ("row", TAGS[0][-4:], TAGS[1][-3:], TAGS[2][-3:], "mean"))
for name in ["vis"] + FUSED:
    vals = [out[t].get(name) for t in TAGS]
    if any(v is None for v in vals):
        continue
    print("%-24s %+10.4f %+10.4f %+10.4f %+10.4f"
          % (name, vals[0], vals[1], vals[2], float(np.mean(vals))))

rows = {name: [out[t][name] for t in TAGS] for name in FUSED if all(out[t].get(name) is not None for t in TAGS)}
json.dump({"tags": TAGS, "rows": rows, "source": {t: out[t]["vis"] for t in TAGS}},
          open(r"D:\CSF-main\ssgf\_restore_seeds_rtdetr_points.json", "w"), indent=2)
print("points written")
