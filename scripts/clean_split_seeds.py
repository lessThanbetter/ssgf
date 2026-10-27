"""Clean-split seeds: per-seed contrasts and a hierarchical bootstrap.

Four seeds of each clean-split detector exist, one reported and three added here. For
each seed the images are resampled 400 times to give that seed's paired interval. The
mean contrast over seeds then gets its own interval from a two-level bootstrap: seeds
are drawn with replacement, and images are drawn within each drawn seed.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_ap import load, pooled  # noqa: E402

PERIMG = r"D:\CSF-main\ssgf\_restore_perimg"
OUT = r"D:\CSF-main\ssgf\_restore_split_seeds.json"
FUSED = ["mean", "activity_only", "tophat", "probe", "tophat_permuted", "probe_permuted",
         "tophat_wshuffle", "probe_wshuffle", "ensemble", "ifcnn", "densefuse", "csf",
         "piafusion", "spgfusion"]
BOOT = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="agn_split")
    ap.add_argument("--seeds", default="_s1,_s2,_s3")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    tags = [a.family] + [a.family + s for s in a.seeds.split(",") if s]
    data, stems = {}, None
    for t in tags:
        p = os.path.join(PERIMG, f"{t}__vis.npz")
        if not os.path.exists(p):
            print("missing", t, flush=True)
            continue
        ref = load(p)
        stems = ref["stems"]
        data[t] = {"ref": ref,
                   "in": {n: load(os.path.join(PERIMG, f"{t}__{n}.npz"))
                          for n in FUSED if os.path.exists(os.path.join(PERIMG, f"{t}__{n}.npz"))}}
    n = len(stems)
    rng = np.random.default_rng(0)
    all_idx = np.arange(n)
    per_seed = {}
    for t in data:
        draws = [rng.choice(all_idx, size=n, replace=True) for _ in range(BOOT)]
        base = np.array([pooled(data[t]["ref"], d)[0] for d in draws])
        per_seed[t] = {}
        for name, st in data[t]["in"].items():
            cur = np.array([pooled(st, d)[0] for d in draws])
            ds = cur - base
            per_seed[t][name] = (float(ds.mean()),
                                 float(np.percentile(ds, 2.5)),
                                 float(np.percentile(ds, 97.5)))
    out = {"tags": list(data), "per_seed": {}}
    print("=" * 20, a.family, "seeds:", list(data))
    print("per-seed contrast of every fused input against the colour visible source")
    for name in FUSED:
        if name not in per_seed[list(data)[0]]:
            continue
        cells = []
        below = above = ns = 0
        for t in data:
            m, lo, hi = per_seed[t][name]
            cells.append("%+.3f [%+.3f,%+.3f]" % (m, lo, hi))
            if hi < 0:
                below += 1
            elif lo > 0:
                above += 1
            else:
                ns += 1
        print("  %-16s %s   below %d above %d unres %d"
              % (name, "  ".join(cells), below, above, ns))
    # two-level bootstrap of the mean over seeds
    print()
    print("two-level bootstrap of the mean over the four seeds (seed and image resampled)")
    refs = {t: data[t]["ref"] for t in data}
    mean_draws = {name: [] for name in FUSED if name in per_seed[list(data)[0]]}
    for _ in range(BOOT):
        picked = [list(data)[i] for i in rng.integers(0, len(data), size=len(data))]
        acc = {k: [] for k in mean_draws}
        for t in picked:
            d = rng.choice(all_idx, size=n, replace=True)
            b = pooled(refs[t], d)[0]
            for name in mean_draws:
                acc[name].append(pooled(data[t]["in"][name], d)[0] - b)
        for name in mean_draws:
            mean_draws[name].append(float(np.mean(acc[name])))
    for name in FUSED:
        if name not in mean_draws:
            continue
        arr = np.array(mean_draws[name])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        star = "*" if (lo > 0 or hi < 0) else " "
        out["per_seed"][name] = {"mean_over_seeds": round(float(arr.mean()), 4),
                                 "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
        print("  %-16s %+.4f [%+.4f,%+.4f]%s" % (name, arr.mean(), lo, hi, star))
    json.dump(out, open(a.out, "w"), indent=2)
    print("SPLIT_SEEDS_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
