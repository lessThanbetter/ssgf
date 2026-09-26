"""Paired bootstrap intervals for the probe-only adaptation.

The point estimates in the paper come from Ultralytics' pooled metric. To keep the
interval on the same estimator, each bootstrap resample pools the per-image
statistics of the resampled images and recomputes that metric.

Usage (from /root/autodl-tmp/ssgf):
    python paired_eval.py --bench llvip --boot 1000
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/ssgf")
from j1_eval2 import match, pooled_metric          # noqa: E402
from robust_eval import FLIR_EVAL, LLVIP_EVAL       # noqa: E402
from robust_eval_common import BASE, FLIR, OUT, RES  # noqa: E402

LLVIP_INPUTS = [
    ("vis", f"{BASE}/datasets/det/test/visible"),
    ("ir", f"{BASE}/datasets/det/test/infrared"),
    ("mean", f"{BASE}/ssgf/data/k1_native/imgs/mean"),
    ("probe", f"{BASE}/ssgf/data/k1_native/imgs/probe"),
    ("e2ecsf", f"{OUT}/e2ecsf_llvip_native"),
    ("dettgt", f"{OUT}/dettgtar_llvip_native"),
    ("robrr", f"{OUT}/robrr_llvip_native"),
    ("proberr", f"{OUT}/proberr_llvip_native"),
    ("probeone", f"{OUT}/probeone_llvip_native"),
    ("probeS1", f"{OUT}/probeS1_llvip_native"),
    ("probeS2", f"{OUT}/probeS2_llvip_native"),
]
FLIR_INPUTS = [
    ("vis", f"{FLIR}/test/visible"),
    ("ir", f"{FLIR}/test/infrared"),
    ("mean", f"{OUT}/flir_mean"),
    ("probe", f"{OUT}/flir_probe"),
    ("robrr", f"{OUT}/flir_robrr"),
    ("proberr", f"{OUT}/flir_proberr"),
    ("probeone", f"{OUT}/flir_probeone"),
    ("probeS1", f"{OUT}/flir_probeS1"),
    ("probeS2", f"{OUT}/flir_probeS2"),
]


def per_image_stats(model_path, inputs, test_root, hw):
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
    pool = {}
    for name, d in inputs:
        stats = []
        for stem in stems:
            p = None
            for ext in (".png", ".jpg"):
                q = os.path.join(d, stem + ext)
                if os.path.exists(q):
                    p = q
                    break
            if p is None:
                continue
            img = np.asarray(Image.open(p).convert("L").resize((W, H)), np.uint8)
            bgr = np.stack([img] * 3, axis=-1)
            r = model.predict(bgr, conf=0.001, imgsz=640, device=0, verbose=False)[0]
            pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            pc = r.boxes.cls.cpu().numpy().astype(np.int64)
            cf = r.boxes.conf.cpu().numpy().astype(np.float32)
            stats.append((match(pb, pc, *gts[stem]), cf, pc, gts[stem][1]))
        pool[name] = stats
    return pool, len(stems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="llvip", choices=["llvip", "flir"])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    if args.bench == "llvip":
        evals, inputs = LLVIP_EVAL, LLVIP_INPUTS
        test_root, hw = f"{BASE}/datasets/det/test", (1024, 1280)
    else:
        evals, inputs = FLIR_EVAL, FLIR_INPUTS
        test_root, hw = f"{FLIR}/test", (512, 640)

    out = {}
    rng = np.random.RandomState(0)
    for ename, epath in evals:
        if not os.path.exists(epath):
            continue
        print(f"== {args.bench} evaluator {ename} ==", flush=True)
        pool, n = per_image_stats(epath, inputs, test_root, hw)
        draws = [rng.randint(0, n, n) for _ in range(args.boot)]
        boot = {}
        for name in pool:
            boot[name] = np.array([pooled_metric(pool[name], d)[0] for d in draws])
            print(f"  {name:10s} mAP50={pooled_metric(pool[name])[0]:.4f}", flush=True)
        out[ename] = {"point": {k: float(pooled_metric(v)[0]) for k, v in pool.items()}}
        for a in ("proberr", "probeS1", "probeS2", "robrr", "probeone", "dettgt"):
            if a not in boot:
                continue
            for b in ("probe", "mean"):
                if b not in boot:
                    continue
                d = boot[a] - boot[b]
                lo, hi = np.percentile(d, [2.5, 97.5])
                sig = "*" if (lo > 0 or hi < 0) else " "
                out[ename][f"{a}-{b}"] = {"delta": float(d.mean()),
                                          "lo": float(lo), "hi": float(hi)}
                print(f"   {a:9s}-{b:6s} d={d.mean():+.4f} "
                      f"[{lo:+.4f},{hi:+.4f}] {sig}", flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/paired_{args.bench}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("PAIRED_DONE", flush=True)


if __name__ == "__main__":
    main()
