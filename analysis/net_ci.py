"""Confidence interval for the network-averaged effect across evaluators.

The quantity of interest is the change in average detection quality across the
whole evaluator set when a fusion rule is adapted, relative to the untuned rule.
Evaluators share no images across benchmarks, so the bootstrap resamples the LLVIP
images once per draw and the FLIR images once per draw, and applies the same
resample to every evaluator of that benchmark. Each evaluator is scored with the
pooled metric used everywhere else in the paper, so the interval belongs to the
same estimator as the point estimates.

Usage (from /root/autodl-tmp/ssgf):
    python net_ci.py --boot 400
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/ssgf")
from j1_eval2 import match, pooled_metric            # noqa: E402
from robust_eval import FLIR_EVAL, LLVIP_EVAL         # noqa: E402
from robust_eval_common import BASE, FLIR, OUT, RES   # noqa: E402

MODELS = {
    "probe": f"{BASE}/ssgf/data/k1_native/imgs/probe",
    "proberr": f"{OUT}/proberr_llvip_native",
    "probeS1": f"{OUT}/probeS1_llvip_native",
    "probeS2": f"{OUT}/probeS2_llvip_native",
    "probeone": f"{OUT}/probeone_llvip_native",
    "robrr": f"{OUT}/robrr_llvip_native",
    "dettgt": f"{OUT}/dettgtar_llvip_native",
    "e2ecsf": f"{OUT}/e2ecsf_llvip_native",
}
FLIR_MODELS = {
    "probe": f"{OUT}/flir_probe",
    "proberr": f"{OUT}/flir_proberr",
    "probeS1": f"{OUT}/flir_probeS1",
    "probeS2": f"{OUT}/flir_probeS2",
    "probeone": f"{OUT}/flir_probeone",
    "robrr": f"{OUT}/flir_robrr",
}


def stats_for(model_path, inputs, test_root, hw, keep):
    from ultralytics import YOLO
    model = YOLO(model_path)
    H, W = hw
    stems = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(f"{test_root}/labels"))

    def gt(stem):
        b, c = [], []
        p = f"{test_root}/labels/{stem}.txt"
        if os.path.exists(p):
            for line in open(p):
                q = line.split()
                if len(q) < 5:
                    continue
                x, y, w, h = map(float, q[1:5])
                b.append([(x - w / 2) * W, (y - h / 2) * H,
                          (x + w / 2) * W, (y + h / 2) * H])
                c.append(0)
        return (np.asarray(b, np.float32) if b else np.zeros((0, 4), np.float32),
                np.asarray(c, np.int64) if c else np.zeros((0,), np.int64))

    gts = {s: gt(s) for s in stems}
    out = {}
    for name in keep:
        d = inputs[name]
        stats = []
        for stem in stems:
            p = None
            for ext in (".png", ".jpg"):
                q = os.path.join(d, stem + ext)
                if os.path.exists(q):
                    p = q
                    break
            if p is None:
                stats.append(None)
                continue
            img = np.asarray(Image.open(p).convert("L").resize((W, H)), np.uint8)
            bgr = np.stack([img] * 3, axis=-1)
            r = model.predict(bgr, conf=0.001, imgsz=640, device=0, verbose=False)[0]
            pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            pc = r.boxes.cls.cpu().numpy().astype(np.int64)
            cf = r.boxes.conf.cpu().numpy().astype(np.float32)
            stats.append((match(pb, pc, *gts[stem]), cf, pc, gts[stem][1]))
        out[name] = stats
        print(f"    collected {name}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.RandomState(0)

    per_bench = {}
    llvip_inputs = {k: v for k, v in MODELS.items()}
    for bench, evals, inputs, test_root, hw in (
            ("llvip", LLVIP_EVAL, llvip_inputs,
             f"{BASE}/datasets/det/test", (1024, 1280)),
            ("flir", FLIR_EVAL, FLIR_MODELS, f"{FLIR}/test", (512, 640))):
        keep = list(inputs)
        got = {}
        for ename, epath in evals:
            if not os.path.exists(epath):
                continue
            print(f"== {bench} evaluator {ename} ==", flush=True)
            got[ename] = stats_for(epath, inputs, test_root, hw, keep)
        n = len(got[[*got][0]]["probe"]) if got else 0
        print(f"[{bench}] {n} images, {len(got)} evaluators", flush=True)
        draws = [rng.randint(0, n, n) for _ in range(args.boot)]
        per_bench[bench] = {"got": got, "draws": draws, "n": n}

    models = sorted({m for b in per_bench.values()
                     for e in b["got"].values() for m in e})
    print("\n== net effect across evaluators ==", flush=True)
    res = {}
    for m in models:
        if m == "probe":
            continue
        deltas = []
        for d in range(args.boot):
            per_eval = []
            for bench, b in per_bench.items():
                idx = b["draws"][d]
                for ename, st in b["got"].items():
                    if m not in st or "probe" not in st:
                        continue
                    per_eval.append(pooled_metric(st[m], idx)[0] -
                                    pooled_metric(st["probe"], idx)[0])
            if per_eval:
                deltas.append(float(np.mean(per_eval)))
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        # point estimate over the same evaluator set
        pts = []
        for bench, b in per_bench.items():
            for ename, st in b["got"].items():
                if m in st:
                    pts.append(pooled_metric(st[m])[0] - pooled_metric(st["probe"])[0])
        res[m] = {"net": float(np.mean(pts)), "lo": float(lo), "hi": float(hi),
                  "n_evaluators": len(pts)}
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {m:9s} net={np.mean(pts):+.4f} [{lo:+.4f},{hi:+.4f}] {star} "
              f"over {len(pts)} evaluators", flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/net_ci.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("NET_CI_DONE", flush=True)


if __name__ == "__main__":
    main()
