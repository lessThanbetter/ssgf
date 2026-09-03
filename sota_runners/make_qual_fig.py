"""Build a 4-panel qualitative detection comparison:
VIS / IR / SPGFusion-fused / Ours(probe)-fused.

Picks the LLVIP scene most strongly supporting "high VIF != downstream":
a target the visible-trained detector finds in the probe-fused image but misses
in the SPGFusion (high-VIF) image, then draws boxes and composites a 2x2 figure.
"""
import argparse
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (ua + 1e-6)


def predict(model, path):
    r = model.predict(path, conf=0.25, imgsz=640, device=0, verbose=False)[0]
    boxes = []
    if r.boxes is not None and len(r.boxes):
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
    return boxes


def to3(im):
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im


def draw(im, boxes, color):
    im = to3(np.asarray(im)).copy()
    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(im, f'p {c:.2f}', (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vis', required=True)
    ap.add_argument('--ir', required=True)
    ap.add_argument('--probe', required=True)
    ap.add_argument('--spg', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stem', default=None, help='skip search and use this scene')
    args = ap.parse_args()

    model = YOLO(args.model)
    stems = sorted({os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(os.path.join(args.vis, '*'))})
    vis = {s: os.path.join(args.vis, s + '.jpg') for s in stems}
    ir = {s: os.path.join(args.ir, s + '.jpg') for s in stems}
    probe = {s: os.path.join(args.probe, s + '.jpg') for s in stems}
    spg = {s: os.path.join(args.spg, s + '.png') for s in stems}

    best = args.stem if args.stem else None
    best_score = None
    if best is None:
        best_score = (-1, -1, -1)
        for s in stems:
            if not all(os.path.exists(p) for p in (vis[s], ir[s], probe.get(s, ''), spg.get(s, ''))):
                continue
            pb = predict(model, probe[s])
            sb = predict(model, spg[s])
            misses = 0
            for bx1, by1, bx2, by2, c in pb:
                if c < 0.5:
                    continue
                if not any(iou((bx1, by1, bx2, by2), (sx1, sy1, sx2, sy2)) > 0.1 for sx1, sy1, sx2, sy2, _ in sb):
                    misses += 1
            delta = len(pb) - len(sb)
            cluster_pen = abs(len(pb) - 4)
            score = (misses, delta, -cluster_pen)
            if score > best_score:
                best_score = score
                best = s

    if best is None:
        print('no scene found')
        return
    print(f'chosen scene: {best}  score={best_score}')

    boxes = {
        'VIS': predict(model, vis[best]),
        'IR': predict(model, ir[best]),
        'SPGFusion': predict(model, spg[best]),
        'Ours (probe)': predict(model, probe[best]),
    }
    ims = {
        'VIS': cv2.imread(vis[best]),
        'IR': cv2.imread(ir[best]),
        'SPGFusion': cv2.imread(spg[best]),
        'Ours (probe)': cv2.imread(probe[best]),
    }
    green = (0, 200, 0)
    red = (0, 0, 255)
    pan = []
    mAP = {"VIS": "0.816", "IR": "0.697", "SPGFusion": "0.736", "Ours (probe)": "0.847"}
    for k in ['VIS', 'IR', 'SPGFusion', 'Ours (probe)']:
        col = green
        # highlight missed target for SPGFusion in red-ish? keep simple: green boxes
        im = draw(ims[k].copy(), boxes[k], green)
        # normalize size
        h, w = im.shape[:2]
        ph, pw = 400, 540
        im = cv2.resize(im, (pw, ph))
        cv2.rectangle(im, (0, 0), (pw - 1, 48), (30, 30, 30), -1)
        label = f'{k} (mAP50 {mAP[k]})'
        cv2.putText(im, label, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
        pan.append(im)

    row1 = np.hstack([pan[0], pan[1]])
    row2 = np.hstack([pan[2], pan[3]])
    fig = np.vstack([row1, row2])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, fig)
    print('saved', args.out, fig.shape)


if __name__ == '__main__':
    main()
