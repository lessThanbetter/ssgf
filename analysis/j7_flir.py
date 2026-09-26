"""J7: repeat the whole experiment on a second detection benchmark.

FLIR-ADAS (aligned, Zhang et al. release) is split into 1500 training and 300 test
pairs that contain at least one person box, matching the LLVIP subset shape. Two
detectors are trained, one on the visible modality only and one on both, and every
fused input is scored with both.

Usage (from /root/autodl-tmp/ssgf):
    python j7_flir.py --task prep
    python j7_flir.py --task train
    python j7_flir.py --task fuse
    python j7_flir.py --task eval
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/ssgf")

BASE = "/root/autodl-tmp"
FLIR = f"{BASE}/datasets/flir"
OUT = f"{BASE}/ssgf/sota_outs"
RUNS = f"{BASE}/ssgf/data/det_runs"
RES = f"{BASE}/ssgf/data/j7_flir"
AE = f"{BASE}/ssgf/data/msrs_ae.pt"
PROBE = f"{BASE}/ssgf/data/msrs_probe.pt"

DETECTORS = {
    "flir_vis": f"{RUNS}/flir_vis/weights/best.pt",
    "flir_agn": f"{RUNS}/flir_agn/weights/best.pt",
}
INPUTS = [
    ("vis", f"{FLIR}/test/visible"),
    ("ir", f"{FLIR}/test/infrared"),
    ("mean", f"{OUT}/flir_mean"),
    ("activity_only", f"{OUT}/flir_activity_only"),
    ("tophat", f"{OUT}/flir_tophat"),
    ("probe", f"{OUT}/flir_probe"),
    ("ensemble", f"{OUT}/flir_ensemble"),
    ("ifcnn", f"{OUT}/flir_ifcnn"),
    ("densefuse", f"{OUT}/flir_densefuse"),
    ("csf", f"{OUT}/flir_csf"),
    ("piafusion", f"{OUT}/flir_pia"),
    ("spgfusion", f"{OUT}/flir_spg"),
]


def stems(split="test"):
    d = os.path.join(FLIR, split, "labels")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d))


def prep():
    """Build the two YOLO training views and their data.yaml files."""
    for name, modalities in (("yolo_vis", ["visible"]), ("yolo_agn", ["visible", "infrared"])):
        root = os.path.join(FLIR, name)
        img = os.path.join(root, "images")
        lab = os.path.join(root, "labels")
        os.makedirs(img, exist_ok=True)
        os.makedirs(lab, exist_ok=True)
        n = 0
        for mod in modalities:
            src = os.path.join(FLIR, "train", mod)
            for f in sorted(os.listdir(src)):
                stem = os.path.splitext(f)[0]
                pref = "v" if mod == "visible" else "i"
                dst = os.path.join(img, f"{pref}_{stem}.jpg")
                if not os.path.exists(dst):
                    os.symlink(os.path.join(src, f), dst)
                dl = os.path.join(lab, f"{pref}_{stem}.txt")
                gl = os.path.join(FLIR, "train", "labels", stem + ".txt")
                if not os.path.exists(dl):
                    os.symlink(gl, dl)
                n += 1
        with open(os.path.join(root, "data.yaml"), "w") as fh:
            fh.write(f"path: {root}\ntrain: images\nval: images\n"
                     f"nc: 1\nnames: {{0: person}}\n")
        print(f"{name}: {n} training images -> {root}/data.yaml", flush=True)


def train():
    from ultralytics import YOLO
    for name in ("yolo_vis", "yolo_agn"):
        out = f"{RUNS}/{name.replace('yolo_', 'flir_')}"
        if os.path.exists(f"{out}/weights/best.pt"):
            print(f"{name}: exists, skip", flush=True)
            continue
        print(f"=== training {name} ===", flush=True)
        m = YOLO(f"{BASE}/ultralytics-main/yolo11n.pt")
        m.train(data=f"{FLIR}/{name}/data.yaml", epochs=40, imgsz=640, batch=16,
                seed=0, device=0, workers=4, verbose=False,
                project=RUNS, name=os.path.basename(out), exist_ok=True)
    print("J7_TRAIN_DONE", flush=True)


def fuse():
    """Fuse the FLIR test pairs with our rules and the two retrained baselines."""
    from ssgf.config import get_default_config
    from ssgf.fusion import MeanRule, ModalityCSRule, SemanticPriorRule, SemanticSaliencyRule
    from ssgf.models import Decoder, Encoder, build_prior
    from ssgf.models.baselines import DenseFuse, IFCNN, fuse_l1
    from ssgf.utils.checkpoint import load_ckpt
    from ssgf.utils.io import imread
try:  # the module ships in this directory; the old path is kept for local runs
    from sota_runners.k1_controls import ActivityOnlyRule
except ImportError:  # pragma: no cover
    from k1_controls import ActivityOnlyRule

    dev = "cuda"
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(AE, encoder=enc, decoder=dec)
    prior = build_prior("tophat")
    mean_r = MeanRule(enc, dec, cfg, dev)
    act_r = ActivityOnlyRule(enc, dec, cfg, dev)
    top_r = SemanticPriorRule(enc, dec, cfg, prior, dev)
    prb_r = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt(PROBE, probe=prb_r.probe)
    csf_r = ModalityCSRule(enc, dec, cfg, dev)
    load_ckpt(f"{BASE}/ssgf/data/msrs_csf.pt", classifier=csf_r.scorer)

    ifc = IFCNN().to(dev)
    ifc.load_state_dict(torch.load(f"{BASE}/ssgf/data/btl/ifcnn.pt", map_location="cpu",
                                   weights_only=False))
    ifc.eval()
    dn = DenseFuse().to(dev)
    dn.load_state_dict(torch.load(f"{BASE}/ssgf/data/btl/densefuse.pt", map_location="cpu",
                                  weights_only=False))
    dn.eval()

    ss = stems()
    print(f"[flir-fuse] {len(ss)} test pairs", flush=True)
    t0 = time.time()
    for k, stem in enumerate(ss):
        v = imread(f"{FLIR}/test/visible/{stem}.jpg")
        r = imread(f"{FLIR}/test/infrared/{stem}.jpg")
        vt = torch.from_numpy(v)[None, None].to(dev)
        rt = torch.from_numpy(r)[None, None].to(dev)
        out = {}
        with torch.no_grad():
            out["mean"] = mean_r.fuse_pair(vt, rt)[0]
            out["activity_only"] = act_r.fuse_pair(vt, rt)[0]
            out["tophat"] = top_r.fuse_pair(vt, rt)[0]
            out["probe"] = prb_r.fuse_pair(vt, rt)[0]
            out["ensemble"] = 0.5 * (out["tophat"] + out["probe"])
            out["ifcnn"] = ((ifc.decode(fuse_l1(ifc.encode(vt * 2 - 1),
                                                ifc.encode(rt * 2 - 1))) + 1) / 2)
            out["densefuse"] = ((dn.decode(fuse_l1(dn.encode(vt * 2 - 1),
                                                  dn.encode(rt * 2 - 1))) + 1) / 2)
        out["csf"] = csf_r.fuse_pair(vt, rt)[0]      # needs grad
        for name, f in out.items():
            d = f"{OUT}/flir_{name}"
            os.makedirs(d, exist_ok=True)
            arr = f[0, 0].detach().clamp(0, 1).cpu().numpy()
            Image.fromarray((arr * 255).astype(np.uint8)).save(f"{d}/{stem}.png")
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(ss)}  {time.time()-t0:.0f}s", flush=True)
    print("J7_FUSE_DONE", flush=True)


def eval_():
    from j1_eval2 import match, pooled_metric
    from ultralytics import YOLO

    def gt(stem):
        b, c = [], []
        p = f"{FLIR}/test/labels/{stem}.txt"
        if os.path.exists(p):
            for line in open(p):
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, x, y, w, h = parts[:5]
                x, y, w, h = map(float, (x, y, w, h))
                b.append([(x - w / 2) * 640, (y - h / 2) * 512,
                          (x + w / 2) * 640, (y + h / 2) * 512])
                c.append(0)
        return (np.asarray(b, np.float32) if b else np.zeros((0, 4), np.float32),
                np.asarray(c, np.int64) if c else np.zeros((0,), np.int64))

    ss = stems()
    gts = {s: gt(s) for s in ss}
    res = {}
    for dname, path in DETECTORS.items():
        if not os.path.exists(path):
            print(f"{dname}: no weights, skip", flush=True)
            continue
        model = YOLO(path)
        pool = {}
        for name, d in INPUTS:
            stats = []
            for stem in ss:
                p = os.path.join(d, stem + ".png")
                if not os.path.exists(p):
                    p = os.path.join(d, stem + ".jpg")
                img = np.asarray(Image.open(p).convert("L").resize((640, 512)), np.uint8)
                bgr = np.stack([img] * 3, axis=-1)
                r = model.predict(bgr, conf=0.001, imgsz=640, device=0, verbose=False)[0]
                pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                pc = r.boxes.cls.cpu().numpy().astype(np.int64)
                cf = r.boxes.conf.cpu().numpy().astype(np.float32)
                stats.append((match(pb, pc, *gts[stem]), cf, pc, gts[stem][1]))
            pool[name] = stats
        res[dname] = {}
        for name, _ in INPUTS:
            m50, m95 = pooled_metric(pool[name])
            res[dname][name] = {"mAP50": m50, "mAP50_95": m95}
            print(f"  {dname:10s} {name:16s} mAP50={m50:.4f}", flush=True)
        rng = np.random.RandomState(0)
        draws = [rng.randint(0, len(ss), len(ss)) for _ in range(500)]
        cache = {n: np.array([pooled_metric(pool[n], d)[0] for d in draws])
                 for n, _ in INPUTS}
        for a, b in (("mean", "vis"), ("probe", "vis"), ("tophat", "vis"),
                     ("activity_only", "vis"), ("piafusion", "vis"),
                     ("spgfusion", "vis"), ("probe", "mean"), ("tophat", "mean"),
                     ("probe", "activity_only"), ("tophat", "activity_only")):
            d = cache[a] - cache[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            star = "*" if (lo > 0 or hi < 0) else " "
            res[dname][f"{a}-{b}"] = {"delta": float(d.mean()), "lo": float(lo),
                                      "hi": float(hi)}
            print(f"  {dname:10s} {a:14s}-{b:14s} d={d.mean():+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] {star}", flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/flir.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("J7_EVAL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["prep", "train", "fuse", "eval"])
    a = ap.parse_args()
    {"prep": prep, "train": train, "fuse": fuse, "eval": eval_}[a.task]()
