"""Score every fused input under the larger visible-trained detector (YOLO11s).

Same estimator and layout as the other runs: per image the predictions are matched to
the ground truth at the ten IoU thresholds with the one-to-one assignment of
`_restore_score`, and the per-image statistics land in `_restore/perimg/yolo11s_vis__*`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/ssgf")

from _restore_score import INPUTS, gt_boxes, match, find  # noqa: E402
from ultralytics import YOLO  # noqa: E402

BASE = "/root/autodl-tmp"
TEST = f"{BASE}/datasets/det/test"
OUT = f"{BASE}/ssgf/_restore/perimg"
NAME = "yolo11s_vis"
CKPT = f"{BASE}/ssgf/data/det_runs/llvip_s/weights/best.pt"


def main():
    if not os.path.exists(CKPT):
        print("no checkpoint yet", flush=True)
        return
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{TEST}/labels"))
    gts = {s: gt_boxes(s) for s in stems}
    n = len(stems)
    n_gt = np.array([sum(1 for _ in open(f"{TEST}/labels/{s}.txt")) for s in stems], np.int32)
    model = YOLO(CKPT)
    print(f"=== {NAME} ===", flush=True)
    for iname, d in INPUTS:
        npz = os.path.join(OUT, f"{NAME}__{iname}.npz")
        if os.path.exists(npz):
            continue
        rows = []
        missing = 0
        for stem in stems:
            p = find(d, stem)
            if p is None:
                rows.append(None)
                missing += 1
                continue
            r = model.predict(p, conf=0.001, imgsz=640, device=0, verbose=False)[0]
            pb = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            pc = r.boxes.cls.cpu().numpy().astype(np.int64)
            cf = r.boxes.conf.cpu().numpy().astype(np.float32)
            gt_b, gt_c = gts[stem]
            rows.append((match(pb, pc, gt_b, gt_c), cf, pc, gt_c))
        if missing:
            print(f"  {iname}: {missing} absent, skipped", flush=True)
            continue
        keep = [r for r in rows if r is not None]
        np.savez_compressed(
            npz,
            tp=np.concatenate([r[0] for r in keep], axis=0),
            conf=np.concatenate([r[1] for r in keep], axis=0),
            pcls=np.concatenate([r[2] for r in keep], axis=0),
            img_idx=np.concatenate([np.full(len(r[1]), i, np.int32)
                                    for i, r in enumerate(rows) if r is not None]),
            tcls=np.concatenate([r[3] for r in keep], axis=0),
            n_gt=n_gt, stems=np.array(stems), n_img=n)
        print(f"  {iname}: written", flush=True)
    print("SCORE_YOLO11S_DONE", flush=True)


if __name__ == "__main__":
    main()
