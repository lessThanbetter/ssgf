"""Detection-side contrast between each targetness rule and its own scrambled version."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_ap import load, pooled  # noqa: E402

PERIMG = r"D:\CSF-main\ssgf\_restore_perimg"
PAIRS = [("tophat", "tophat_permuted"), ("tophat", "tophat_wshuffle"),
         ("probe", "probe_permuted"), ("probe", "probe_wshuffle")]
DETS = ["yolo11n_vis", "agnostic"]
DRAWS = 1000


def main():
    out = {}
    rng = np.random.default_rng(3)
    for det in DETS:
        ref = load(os.path.join(PERIMG, "%s__vis.npz" % det))
        n = ref["n"]
        idx = [rng.choice(n, size=n, replace=True) for _ in range(DRAWS)]
        out[det] = {}
        for a, b in PAIRS:
            pa = os.path.join(PERIMG, "%s__%s.npz" % (det, a))
            pb = os.path.join(PERIMG, "%s__%s.npz" % (det, b))
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            sa, sb = load(pa), load(pb)
            d = np.array([pooled(sa, dd)[0] - pooled(sb, dd)[0] for dd in idx])
            out[det]["%s-%s" % (a, b)] = {
                "delta": float(d.mean()),
                "lo": float(np.percentile(d, 2.5)),
                "hi": float(np.percentile(d, 97.5))}
            print("  %-14s %-28s %+.4f [%+.4f,%+.4f]"
                  % (det, "%s - %s" % (a, b), d.mean(),
                     np.percentile(d, 2.5), np.percentile(d, 97.5)))
    json.dump(out, open(r"D:\CSF-main\ssgf\_restore_scramble.json", "w"), indent=2)
    print("SCRAMBLE_DONE")


if __name__ == "__main__":
    main()
