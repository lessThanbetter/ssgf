"""SPGFusion inference on arbitrary IR/VIS pairs, output grayscale fused images.

Reuses the official Fusion_dataset transforms (resize to multiples of 16, RGB).
Only the luminance channel of the fused RGB is saved (no min-max stretch).

Usage:
    cd /root/autodl-tmp/ssgf/sota/SPGFusion
    python /root/autodl-tmp/ssgf/sota_runners/run_spgfusion.py \
        --ir  <ir_dir> --vis <vis_dir> --out <out_dir>
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from device import device
from model.SPGFusion import SPGFusion
from model.Dino_Clip import DINOiser


def luma_to_u8(img):  # img: [H,W] float in [0,1]
    img = img.clamp(0.0, 1.0).cpu().numpy()
    return (img * 255.0).astype(np.uint8)


def build_pairs(ir_dir, vis_dir):
    """Return ordered list of (vis_path, ir_path, name) matched by stem."""
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
    vis = [p for e in exts for p in glob.glob(os.path.join(vis_dir, e))]
    ir = [p for e in exts for p in glob.glob(os.path.join(ir_dir, e))]
    vis_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in vis}
    ir_by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in ir}
    common = sorted(set(vis_by_stem) & set(ir_by_stem))
    if common:
        return [(vis_by_stem[s], ir_by_stem[s], s) for s in common]
    # fallback: index pairing
    vis.sort(); ir.sort()
    return [(vis[i], ir[i], os.path.splitext(os.path.basename(vis[i]))[0])
            for i in range(min(len(vis), len(ir)))]


def load_pair(vis_path, ir_path, max_side=512):
    vis = Image.open(vis_path).convert('RGB')
    ir = Image.open(ir_path).convert('RGB')
    w, h = vis.size
    scale = max_side / max(w, h)
    nw = max(16, (int(w * scale) // 16) * 16)
    nh = max(16, (int(h * scale) // 16) * 16)
    vis = vis.resize((nw, nh), Image.BICUBIC)
    ir = ir.resize((nw, nh), Image.BICUBIC)
    return TF.to_tensor(vis).unsqueeze(0), TF.to_tensor(ir).unsqueeze(0), (nw, nh)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ir', required=True)
    p.add_argument('--vis', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--config', default='clip_dinoiser.yaml')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--limit', type=int, default=0, help='run only first N pairs (0 = all)')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('[spg] building DINO+CLIP semantic extractor ...')
    clip_cfg = {
        'clip_model': 'ViT-B/16',
        'clip_model_pth': 'model/ViT-B-16-laion2B/open_clip_pytorch_model.bin',
        'patch_size': 16,
        'img_size': 224,
    }
    model_ex = DINOiser(clip_cfg).to(device)
    model_ex.load_dino()  # triggers torch.hub download of dino_vitbase16 if absent
    model_ex.to(device)
    model_ex.eval()

    print('[spg] loading fusion network ...')
    fus = SPGFusion().to(device)
    ck = torch.load('CHECKPOINT/best_model_1.pth', map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    fus.load_state_dict(sd, strict=True)
    fus.eval()

    pairs = build_pairs(args.ir, args.vis)
    n = len(pairs)
    if args.limit:
        n = min(n, args.limit)
    print(f'[spg] {n} pairs -> {args.out}')

    done = 0
    with torch.no_grad():
        for i, (vis_path, ir_path, stem) in enumerate(pairs):
            if done >= n:
                break
            vis, ir, _ = load_pair(vis_path, ir_path)
            vis = vis.to(device)
            ir = ir.to(device)
            vi_sem = model_ex(vis)
            ir_sem = model_ex(ir)
            fused = fus(vis, ir, vi_sem, ir_sem)  # [B,3,H,W]
            im = fused[0]
            gray = 0.299 * im[0] + 0.587 * im[1] + 0.114 * im[2]
            cv2.imwrite(os.path.join(args.out, stem + '.png'), luma_to_u8(gray))
            done += 1
            if done % 10 == 0:
                print(f'[spg] {done}/{n}')
    print(f'[spg] finished, saved {done} images')


if __name__ == '__main__':
    main()
