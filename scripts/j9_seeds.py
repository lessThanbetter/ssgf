"""J9: seed replication of a second architecture, and a third FLIR evaluator.

Part 1 trains YOLO26n on the same 1500 visible LLVIP images with seeds 1 to 3, so
that the effect of changing the detector architecture can be compared with the
effect of changing the random seed inside one architecture.

Part 2 trains YOLO26n on the same 1500 FLIR visible images, giving the second
benchmark a third evaluator and matching the structure of the LLVIP comparison.

Usage (from /root/autodl-tmp/ssgf):
    python j9_seeds.py --task train
    python j9_seeds.py --task eval
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
RUNS = f"{BASE}/ssgf/data/det_runs"
OUT = f"{BASE}/ssgf/sota_outs"
FLIR = f"{BASE}/datasets/flir"
RES = f"{BASE}/ssgf/data/j9_seeds"

LLVIP_INPUTS = [
    ("vis", f"{TEST}/visible"),
    ("ir", f"{TEST}/infrared"),
    ("mean", f"{BASE}/ssgf/data/k1_native/imgs/mean"),
    ("activity_only", f"{BASE}/ssgf/data/k1_native/imgs/activity_only"),
    ("tophat", f"{BASE}/ssgf/data/k1_native/imgs/tophat"),
    ("probe", f"{BASE}/ssgf/data/k1_native/imgs/probe"),
    ("ensemble", f"{OUT}/ensemble_llvip_native"),
    ("ifcnn", f"{OUT}/ifcnn_llvip_native"),
    ("densefuse", f"{OUT}/densefuse_llvip_native"),
    ("csf", f"{OUT}/csf_llvip_native"),
    ("piafusion", f"{OUT}/pia_llvip_native"),
    ("spgfusion", f"{OUT}/spg_llvip_native"),
]
FLIR_INPUTS = [
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


def train():
    from ultralytics import YOLO
    jobs = [(f"llvip_y26_seed{s}", f"{BASE}/datasets/det/yolo.yaml", s)
            for s in (1, 2, 3)]
    jobs.append(("flir_y26", f"{FLIR}/yolo_vis/data.yaml", 0))
    for name, data, seed in jobs:
        if os.path.exists(f"{RUNS}/{name}/weights/best.pt"):
            print(f"{name}: exists, skip", flush=True)
            continue
        print(f"=== training {name} (seed {seed}) ===", flush=True)
        m = YOLO(f"{BASE}/ssgf/yolo26n.pt")
        m.train(data=data, epochs=40, imgsz=640, batch=16, seed=seed, device=0,
                workers=4, verbose=False, project=RUNS, name=name, exist_ok=True)
    print("J9_TRAIN_DONE", flush=True)


def evaluate_inputs(model_path, inputs, test_root, img_hw, tag):
    from ultralytics import YOLO
    model = YOLO(model_path)
    stems = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(f"{test_root}/labels"))
    H, W = img_hw

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
    for name, d in inputs:
        stats = []
        for stem in stems:
            p = find_image(d, stem)
            if p is None:
                continue
            img = np.asarray(Image.open(p).convert("L").resize((W, H)), np.uint8)
            bgr = np.stack([img] * 3, axis=-1)
            r = model.predict(bgr, conf=0.001, imgsz=640, device=0, verbose=False)[0]
            pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            pc = r.boxes.cls.cpu().numpy().astype(np.int64)
            cf = r.boxes.conf.cpu().numpy().astype(np.float32)
            stats.append((match(pb, pc, *gts[stem]), cf, pc, gts[stem][1]))
        m50, m95 = pooled_metric(stats)
        out[name] = {"mAP50": m50, "mAP50_95": m95}
        print(f"  {tag} {name:14s} mAP50={m50:.4f}", flush=True)
    return out


def eval_():
    os.makedirs(RES, exist_ok=True)
    res = {"llvip_y26_seeds": {}, "flir_y26": None}

    # --- LLVIP: three more seeds of the second architecture ---
    for s in (1, 2, 3):
        p = f"{RUNS}/llvip_y26_seed{s}/weights/best.pt"
        if not os.path.exists(p):
            print(f"missing {p}", flush=True)
            continue
        print(f"== YOLO26n seed {s} ==", flush=True)
        res["llvip_y26_seeds"][f"seed{s}"] = evaluate_inputs(
            p, LLVIP_INPUTS, f"{BASE}/datasets/det/test", (1024, 1280),
            f"y26s{s}")

    # --- FLIR: a third evaluator, same architecture family as the fourth LLVIP one ---
    p = f"{RUNS}/flir_y26/weights/best.pt"
    if os.path.exists(p):
        print("== FLIR YOLO26n ==", flush=True)
        res["flir_y26"] = evaluate_inputs(p, FLIR_INPUTS, f"{FLIR}/test",
                                          (512, 640), "flir_y26")
    with open(f"{RES}/j9.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("J9_EVAL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["train", "eval"])
    a = ap.parse_args()
    train() if a.task == "train" else eval_()
