"""Bootstrap 95% CI for the visible-trained-detector mAP50 of mean/top-hat/probe."""
import os
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, build_prior
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (ua + 1e-9)


def ap50(dets, gts, iou_t=0.5):
    """dets: list of (image_id, conf, box). gts: {image_id: [boxes]}."""
    order = np.argsort([-d[1] for d in dets])
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    used = {k: [False] * len(v) for k, v in gts.items()}
    for i in order:
        img, conf, box = dets[i]
        best = -1; best_i = -1; best_iou = -1
        for j, g in enumerate(gts.get(img, [])):
            if used[img][j]:
                continue
            v = iou(box, g)
            if v > best_iou:
                best_iou = v; best = j; best_i = j
        if best_i >= 0 and best_iou >= iou_t:
            tp[i] = 1; used[img][best_i] = True
        else:
            fp[i] = 1
    fp = np.cumsum(fp); tp = np.cumsum(tp)
    rec = tp / max(sum(len(v) for v in gts.values()), 1)
    prec = tp / np.maximum(tp + fp, 1e-9)
    # 101-point interpolated precision
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        idx = np.where(rec >= t)[0]
        ap += max(prec[idx]) if len(idx) else 0
    return ap / 101


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_ae.pt', encoder=enc, decoder=dec)
    prior = build_prior('tophat')
    mean_rule = MeanRule(enc, dec, cfg, dev)
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_probe.pt', probe=probe_rule.probe)
    det = YOLO('/root/autodl-tmp/ssgf/data/det_runs/llvip/weights/best.pt')

    test = '/root/autodl-tmp/datasets/det/test'
    names = sorted(os.listdir(f'{test}/labels'))
    # cache per-image dets for each method
    per = {'mean': [], 'top-hat': [], 'probe': []}
    gts = {}
    for lab in names:
        img_id = lab.replace('.txt', '.jpg')
        vis = imread(f'{test}/visible/{img_id}'); ir = imread(f'{test}/infrared/{img_id}')
        h, w = vis.shape
        sc = 512 / max(h, w)
        nw, nh = w, h
        if sc < 1:
            nw, nh = int(w * sc), int(h * sc)
            vis = np.asarray(Image.fromarray((vis * 255).astype('uint8')).resize((nw, nh)), np.float32) / 255.0
            ir = np.asarray(Image.fromarray((ir * 255).astype('uint8')).resize((nw, nh)), np.float32) / 255.0
        v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
        i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)
        # gt
        boxes = []
        for line in open(f'{test}/labels/{lab}'):
            x, y, bw, bh = map(float, line.split()[1:5])
            x1, y1, x2, y2 = (x - bw / 2) * nw, (y - bh / 2) * nh, (x + bw / 2) * nw, (y + bh / 2) * nh
            boxes.append((x1, y1, x2, y2))
        gts[img_id] = boxes
        with torch.no_grad():
            fmean = mean_rule.fuse_pair(v, i)[0].squeeze().cpu().numpy()
            ftop = top_rule.fuse_pair(v, i)[0].squeeze().cpu().numpy()
            fprobe = probe_rule.fuse_pair(v, i)[0].squeeze().cpu().numpy()
        for m, f in [('mean', fmean), ('top-hat', ftop), ('probe', fprobe)]:
            fu = ((np.clip(f, 0, 1) * 255).astype(np.uint8))
            fu = cv2.cvtColor(fu, cv2.COLOR_GRAY2BGR)
            r = det.predict(fu, conf=0.25, imgsz=640, device=0, verbose=False)[0]
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                per[m].append((img_id, float(c), tuple(b.tolist())))

    def point(m):
        return ap50(per[m], gts)
    rng = np.random.RandomState(0)
    ids = list(gts.keys())
    B = 200
    boot = {m: [] for m in per}
    for b in range(B):
        res = list(rng.choice(ids, size=len(ids), replace=True))
        for m in per:
            dets = [d for d in per[m] if d[0] in set(res)]
            gg = {k: gts[k] for k in res}
            boot[m].append(ap50(dets, gg))
    for m in per:
        arr = np.array(boot[m])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f'{m:8s} mAP50={point(m):.4f}  95% CI [{lo:.4f}, {hi:.4f}]')
    pm = np.array(boot['probe']) - np.array(boot['mean'])
    lo, hi = np.percentile(pm, [2.5, 97.5])
    print(f'probe-mean diff 95% CI [{lo:.4f}, {hi:.4f}]')


if __name__ == '__main__':
    main()
