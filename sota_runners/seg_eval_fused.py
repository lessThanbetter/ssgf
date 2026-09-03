"""Evaluate a fixed U-Net (MSRS-train) on pre-fused MSRS-test images -> mIoU."""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from ssgf.models.segnet import UNet
from seg_downstream import N_CLASSES, SIZE, _mIoU


def eval_metric(model, img, gt, dev):
    with torch.no_grad():
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(dev)
        x = torch.nn.functional.interpolate(x, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
        logits = model(x)
        pred = logits.argmax(1)[0].cpu().numpy()
    gt_r = np.asarray(Image.fromarray(gt.astype('uint8')).resize((SIZE, SIZE), Image.NEAREST), np.int64)
    return _mIoU(pred, gt_r, N_CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fuse', required=True)
    ap.add_argument('--msrs', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    seg = UNet(in_ch=1, n_classes=N_CLASSES).to(dev)
    seg.load_state_dict(torch.load(args.model, map_location='cpu'))
    seg.eval()

    lab_dir = os.path.join(args.msrs, 'test', 'Segmentation_labels')
    vals = []
    for p in glob.glob(os.path.join(args.fuse, '*.png')):
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = os.path.join(lab_dir, stem + '.png')
        if not os.path.exists(lab):
            continue
        img = np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(lab), dtype=np.int64)
        vals.append(eval_metric(seg, img, gt, dev))
    vals = np.array(vals)
    print(f'[{args.tag}] n={len(vals)}  mIoU={vals.mean():.4f} +/- {vals.std():.4f}')


if __name__ == '__main__':
    main()
