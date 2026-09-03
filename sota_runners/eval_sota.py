"""Compute the SSGF metric suite over a baseline's fused images.

Sources are resized to the fused image size before metric computation so VIF /
SSIM compare same-resolution content. Prints means over the dataset.

Usage (from /root/autodl-tmp/ssgf so the `ssgf` package is importable):
    python sota_runners/eval_sota.py \
        --fuse <fused_dir> --vis <vis_dir> --ir <ir_dir> --tag <name>
"""
import argparse
import glob
import os

import cv2
import numpy as np

from ssgf.utils.metrics import report


def load_gray(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return im.astype(np.float32) / 255.0


def index(dirp, exts=('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')):
    d = {}
    for e in exts:
        for p in glob.glob(os.path.join(dirp, e)):
            d[os.path.splitext(os.path.basename(p))[0]] = p
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fuse', required=True)
    ap.add_argument('--vis', required=True)
    ap.add_argument('--ir', required=True)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    vis = index(args.vis)
    ir = index(args.ir)
    names = []
    for e in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif'):
        names += [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.fuse, e))]
    names = sorted(set(names))

    acc = None
    n = 0
    for stem in names:
        fp = os.path.join(args.fuse, stem + '.png')
        if not os.path.exists(fp):
            continue
        if stem not in vis or stem not in ir:
            continue
        f = load_gray(fp)
        a = load_gray(vis[stem])
        b = load_gray(ir[stem])
        a = cv2.resize(a, (f.shape[1], f.shape[0]))
        b = cv2.resize(b, (f.shape[1], f.shape[0]))
        d = report(f, a, b)
        if acc is None:
            acc = {k: [] for k in d}
        for k, v in d.items():
            acc[k].append(v)
        n += 1

    if n == 0:
        print(f'[{args.tag}] no matched images')
        return
    keys = ['VIF', 'EN', 'SD', 'SF', 'AG', 'Qabf', 'SCD', 'SSIM_vis', 'SSIM_ir', 'PSNR_vis', 'PSNR_ir']
    print(f'[{args.tag}] n={n}')
    for k in keys:
        if k in acc:
            arr = np.array(acc[k])
            print(f'  {k:10s} {arr.mean():.4f} +/- {arr.std():.4f}')
    # JSON for the table
    out = {k: round(float(np.mean(acc[k])), 4) for k in keys if k in acc}
    out['n'] = n
    import json as _json
    print(_json.dumps(out))


if __name__ == '__main__':
    main()
