"""J6: a second and third detector family for the evaluator-dependence claim.

The central claim is that the ranking of fusion rules belongs to the evaluator.
The two networks already used are both YOLO11n trained on the same images, so a
referee can read the result as an effect of training modality rather than of the
detector. This script adds two further evaluators trained on the same 1500 visible
images: YOLO26n, a different generation of the same family, and RT-DETR-l, a
transformer detector with a different assignment rule.

Usage (from /root/autodl-tmp/ssgf):
    python j6_det_family.py --task train
    python j6_det_family.py --task eval
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/ssgf")
from j1_eval2 import TEST, find_image, gt_boxes, match, pooled_metric  # noqa: E402

BASE = "/root/autodl-tmp"
OUTDIR = f"{BASE}/ssgf/data/j6_family"
DATA = f"{BASE}/datasets/det/yolo.yaml"
INPUTS = [
    ("vis", f"{TEST}/visible"),
    ("ir", f"{TEST}/infrared"),
    ("mean", f"{BASE}/ssgf/data/k1_native/imgs/mean"),
    ("activity_only", f"{BASE}/ssgf/data/k1_native/imgs/activity_only"),
    ("tophat", f"{BASE}/ssgf/data/k1_native/imgs/tophat"),
    ("probe", f"{BASE}/ssgf/data/k1_native/imgs/probe"),
    ("ensemble", f"{BASE}/ssgf/sota_outs/ensemble_llvip_native"),
    ("ifcnn", f"{BASE}/ssgf/sota_outs/ifcnn_llvip_native"),
    ("densefuse", f"{BASE}/ssgf/sota_outs/densefuse_llvip_native"),
    ("csf", f"{BASE}/ssgf/sota_outs/csf_llvip_native"),
    ("piafusion", f"{BASE}/ssgf/sota_outs/pia_llvip_native"),
    ("spgfusion", f"{BASE}/ssgf/sota_outs/spg_llvip_native"),
]

JOBS = [
    {
        "name": "yolo26n_vis",
        "kind": "yolo",
        "weights": f"{BASE}/ssgf/yolo26n.pt",
        "epochs": 40,
        "batch": 16,
        "out": f"{BASE}/ssgf/data/det_runs/llvip_y26",
    },
    {
        "name": "rtdetr_l_vis",
        "kind": "rtdetr",
        "weights": f"{BASE}/ssgf/rtdetr-l.pt",
        "epochs": 30,
        "batch": 8,
        "out": f"{BASE}/ssgf/data/det_runs/llvip_rtdetr",
    },
]


def train():
    from ultralytics import RTDETR, YOLO
    for j in JOBS:
        best = f"{j['out']}/weights/best.pt"
        if os.path.exists(best):
            print(f"{j['name']}: exists, skip", flush=True)
            continue
        print(f"=== training {j['name']} ===", flush=True)
        model = (YOLO(j["weights"]) if j["kind"] == "yolo"
                 else RTDETR(j["weights"]))
        model.train(data=DATA, epochs=j["epochs"], imgsz=640,
                    batch=j["batch"], seed=0, device=0, workers=4,
                    verbose=False, project=os.path.dirname(j["out"]),
                    name=os.path.basename(j["out"]), exist_ok=True)
    print("J6_TRAIN_DONE", flush=True)


def evaluate(boot=500):
    from ultralytics import RTDETR, YOLO
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{TEST}/labels"))
    gts = {s: gt_boxes(s) for s in stems}
    models = {"yolo11n_vis": f"{BASE}/ssgf/data/det_runs/llvip/weights/best.pt"}
    for j in JOBS:
        p = f"{j['out']}/weights/best.pt"
        if os.path.exists(p):
            models[j["name"]] = p
    res = {}
    for mname, path in models.items():
        model = (YOLO(path) if "rtdetr" not in mname else RTDETR(path))
        pool = {name: [] for name, _ in INPUTS}
        for k, stem in enumerate(stems):
            gt_b, gt_c = gts[stem]
            for name, d in INPUTS:
                p = find_image(d, stem)
                if p is None:
                    pool[name].append(None)
                    continue
                img = np.asarray(Image.open(p).convert("L").resize((1280, 1024)),
                                 np.uint8)
                bgr = np.stack([img] * 3, axis=-1)
                r = model.predict(bgr, conf=0.001, imgsz=640, device=0,
                                  verbose=False)[0]
                pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                pc = r.boxes.cls.cpu().numpy().astype(np.int64)
                cf = r.boxes.conf.cpu().numpy().astype(np.float32)
                pool[name].append((match(pb, pc, gt_b, gt_c), cf, pc, gt_c))
            if (k + 1) % 100 == 0:
                print(f"  [{mname}] {k+1}/{len(stems)}", flush=True)
        res[mname] = {}
        for name, _ in INPUTS:
            stats = [s for s in pool[name] if s is not None]
            m50, m95 = pooled_metric(stats)
            res[mname][name] = {"mAP50": m50, "mAP50_95": m95}
            print(f"  {mname:16s} {name:16s} mAP50={m50:.4f}", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(f"{OUTDIR}/family.json", "w") as fh:
        json.dump(res, fh, indent=2)
    # rank of each rule under each evaluator, to expose the re-ranking
    print("== ranking by mAP50 ==", flush=True)
    for mname, sc in res.items():
        order = sorted(sc.items(), key=lambda kv: -kv[1]["mAP50"])
        print(f"  {mname}: " + " > ".join(f"{k}({v['mAP50']:.3f})"
                                           for k, v in order), flush=True)
    print("J6_EVAL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="train", choices=["train", "eval"])
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    train() if a.task == "train" else evaluate(a.boot)
