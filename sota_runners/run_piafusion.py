"""PIAFusion (PyTorch port) inference -> grayscale fused images.

The fusion is performed in the Y (luminance) channel; we output that Y channel
directly as the grayscale fused image (no min-max stretch).

Usage (from PIAFusion repo root):
    python /root/autodl-tmp/ssgf/sota_runners/run_piafusion.py \
        --ir <ir_dir> --vis <vis_dir> --out <out_dir>
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from models.common import RGB2YCrCb, clamp
from models.fusion_model import PIAFusion


def build_pairs(ir_dir, vis_dir):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
    vis = [p for e in exts for p in glob.glob(os.path.join(vis_dir, e))]
    ir = [p for e in exts for p in glob.glob(os.path.join(ir_dir, e))]
    vis_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in vis}
    ir_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in ir}
    common = sorted(set(vis_by_stem) & set(ir_by_stem))
    if common:
        return [(vis_by_stem[s], ir_by_stem[s], s) for s in common]
    vis.sort(); ir.sort()
    return [(vis[i], ir[i], os.path.splitext(os.path.basename(vis[i]))[0])
            for i in range(min(len(vis), len(ir)))]


def load_pair(vis_path, ir_path, max_side=512):
    vis = Image.open(vis_path).convert('RGB')
    ir = Image.open(ir_path).convert('L')
    w, h = vis.size
    scale = max_side / max(w, h)
    nw = max(16, (int(w * scale) // 16) * 16)
    nh = max(16, (int(h * scale) // 16) * 16)
    vis = vis.resize((nw, nh), Image.BICUBIC)
    ir = ir.resize((nw, nh), Image.BICUBIC)
    return TF.to_tensor(vis), TF.to_tensor(ir)  # [3,H,W], [1,H,W]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ir', required=True)
    p.add_argument('--vis', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print('[pia] loading fusion model ...')
    model = PIAFusion().cuda()
    sd = torch.load('pretrained/fusion_model_epoch_29.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model.eval()

    pairs = build_pairs(args.ir, args.vis)
    n = len(pairs)
    if args.limit:
        n = min(n, args.limit)
    print(f'[pia] {n} pairs -> {args.out}')

    done = 0
    with torch.no_grad():
        for vis_path, ir_path, stem in pairs:
            if done >= n:
                break
            vis_t, ir_t = load_pair(vis_path, ir_path)
            vis_t = vis_t.cuda()
            ir_t = ir_t.cuda()
            y, _cb, _cr = RGB2YCrCb(vis_t)          # [1,H,W]
            vis_y = y.unsqueeze(0)                  # [1,1,H,W]
            inf = ir_t.unsqueeze(0)                 # [1,1,H,W]
            fused = clamp(model(vis_y, inf))        # [1,1,H,W]
            gray = fused[0, 0].clamp(0, 1).cpu().numpy()
            cv2.imwrite(os.path.join(args.out, stem + '.png'),
                        (gray * 255.0).astype(np.uint8))
            done += 1
            if done % 10 == 0:
                print(f'[pia] {done}/{n}')
    print(f'[pia] finished, saved {done} images')


if __name__ == '__main__':
    main()
