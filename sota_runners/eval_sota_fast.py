"""Fast SSGF metric eval for SOTA baselines: VIF + cheap image metrics.

Skips the expensive Qabf / SCD / MS-SSIM. VIF is computed exactly as
`metrics.report` (mean of VIF_vis and VIF_ir).
"""
import argparse
import glob
import os

import cv2
import numpy as np

from ssgf.utils.metrics import (
    average_gradient,
    entropy,
    psnr,
    spatial_frequency,
    ssim,
    std,
    vifp,
)


def load_gray(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


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
    stems = sorted({os.path.splitext(os.path.basename(p))[0]
                    for e in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
                    for p in glob.glob(os.path.join(args.fuse, e))})

    acc = None
    n = 0
    for stem in stems:
        fp = os.path.join(args.fuse, stem + '.png')
        if not os.path.exists(fp) or stem not in vis or stem not in ir:
            continue
        f = load_gray(fp)
        a = cv2.resize(load_gray(vis[stem]), (f.shape[1], f.shape[0]))
        b = cv2.resize(load_gray(ir[stem]), (f.shape[1], f.shape[0]))
        v_vis = vifp(a, f)
        v_ir = vifp(b, f)
        d = {
            'VIF': (v_vis + v_ir) / 2,
            'EN': entropy(f),
            'SD': std(f),
            'SF': spatial_frequency(f),
            'AG': average_gradient(f),
            'SSIM_vis': ssim(f, a),
            'SSIM_ir': ssim(f, b),
            'PSNR_vis': psnr(f, a),
            'PSNR_ir': psnr(f, b),
        }
        if acc is None:
            acc = {k: [] for k in d}
        for k, v in d.items():
            acc[k].append(v)
        n += 1

    if n == 0:
        print(f'[{args.tag}] no matched images')
        return
    print(f'[{args.tag}] n={n}')
    out = {}
    for k in ['VIF', 'EN', 'SD', 'SF', 'AG', 'SSIM_vis', 'SSIM_ir', 'PSNR_vis', 'PSNR_ir']:
        arr = np.array(acc[k])
        mean = arr.mean()
        out[k] = round(float(mean), 4)
        print(f'  {k:10s} {mean:.4f} +/- {arr.std():.4f}')
    out['n'] = n
    import json
    print(json.dumps(out))


if __name__ == '__main__':
    main()
