"""Does the fidelity ranking survive the grid and the storage of the fused image?

Same 30-image protocol as Supplementary Table S6, same eight variants, reported index
configuration, four conditions: the reported 400x512 grid, a half grid, a double grid,
and the reported grid after JPEG quality 50.
"""
import importlib.util
import io
import json
import os

import numpy as np
from PIL import Image

SRC = r"D:\CSF-main\ssgf\_srv_vif_decomp2.py"
spec = importlib.util.spec_from_file_location("vifdecomp", SRC)
vd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd)

DEP = r"D:\CSF-main\ssgf\_deposit\ssgf-llvip-fusion-detection-v1\data\fused_images"
VIS = r"D:\CSF-main\ssgf\data\det_subset\test\visible"
IR = r"D:\CSF-main\ssgf\data\det_subset\test\infrared"
VARIANTS = [
    ("mean", os.path.join(DEP, "datasets", "det", "eval_fused", "mean", "images")),
    ("tophat", os.path.join(DEP, "datasets", "det", "eval_fused", "tophat", "images")),
    ("probe", os.path.join(DEP, "datasets", "det", "eval_fused", "probe", "images")),
    ("ensemble", os.path.join(DEP, "datasets", "det", "eval_fused", "ensemble", "images")),
    ("ifcnn", os.path.join(DEP, "datasets", "det", "eval_fused", "ifcnn", "images")),
    ("densefuse", os.path.join(DEP, "datasets", "det", "eval_fused", "densefuse", "images")),
    ("piafusion", os.path.join(DEP, "ssgf", "sota_outs", "pia_llvip_native_fp16")),
    ("spgfusion", os.path.join(DEP, "ssgf", "sota_outs", "spg_llvip_native_fp16")),
]
CONDITIONS = [("reported 400x512", (400, 512), None),
              ("half grid 200x256", (200, 256), None),
              ("double grid 800x1024", (800, 1024), None),
              ("jpeg50 at 400x512", (400, 512), 50)]
LIMIT = 30
CFG = vd.CONFIGS[0]           # the reported index configuration


def jpeg(img, quality):
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).save(
        buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"), dtype=np.float32) / 255.0


def main():
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(VIS))
    keep = []
    for s in stems:
        if vd.find(VIS, s) and vd.find(IR, s) and all(vd.find(d, s) for _, d in VARIANTS):
            keep.append(s)
        if len(keep) >= LIMIT:
            break
    print("usable stems:", len(keep), flush=True)

    acc = {(n, c[0]): [] for n, _ in VARIANTS for c in CONDITIONS}
    for i, stem in enumerate(keep):
        for name, d in VARIANTS:
            fp = vd.find(d, stem)
            for cname, target, quality in CONDITIONS:
                vis, ir, fu = vd._load(vd.find(VIS, stem), vd.find(IR, stem), fp, target)
                if quality is not None:
                    fu = jpeg(fu, quality)
                acc[(name, cname)].append(vd.score(vis, ir, fu, CFG))
        print("[fidcond] %d/%d" % (i + 1, len(keep)), flush=True)

    rows = []
    for name, _ in VARIANTS:
        row = {"variant": name}
        for cname, _t, _q in CONDITIONS:
            row[cname] = float(np.mean(acc[(name, cname)]))
        rows.append(row)

    out = {"limit": LIMIT, "rows": rows}
    json.dump(out, open(r"D:\CSF-main\ssgf\_restore_fidelity_conditions.json", "w"), indent=2)
    print()
    cols = [c[0] for c in CONDITIONS]
    print("%-12s %s" % ("variant", " ".join("%18s" % c for c in cols)))
    for r in rows:
        print("%-12s %s" % (r["variant"], " ".join("%18.4f" % r[c] for c in cols)))
    base = {r["variant"]: r[cols[0]] for r in rows}
    for c in cols:
        order = [r["variant"] for r in sorted(rows, key=lambda x: -x[c])]
        span = max(r[c] for r in rows) - min(r[c] for r in rows)
        print("ranking %-20s %s (span %.3f)" % (c, " > ".join(order), span))
    print("FIDCOND_DONE")


if __name__ == "__main__":
    main()
