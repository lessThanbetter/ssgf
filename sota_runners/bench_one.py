"""Standalone latency/params for one SOTA method. Usage: bench_one.py pia|spg"""
import os
import sys
import time

import torch


def run(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def pia():
    from models.fusion_model import PIAFusion
    m = PIAFusion().cuda().eval()
    m.load_state_dict(torch.load('pretrained/fusion_model_epoch_29.pth',
                                 map_location='cpu', weights_only=False))
    v = torch.randn(1, 1, 384, 512, device='cuda')
    i = torch.randn(1, 1, 384, 512, device='cuda')
    n = sum(p.numel() for p in m.parameters()) / 1e6
    def fn():
        with torch.no_grad():
            m(v, i)
    print('PIAFusion(512)  %.1f ms  params=%.2fM' % (run(fn, 20), n))


def spg():
    from model.SPGFusion import SPGFusion
    from model.Dino_Clip import DINOiser
    clip_cfg = {'clip_model': 'ViT-B/16',
                'clip_model_pth': 'model/ViT-B-16-laion2B/open_clip_pytorch_model.bin',
                'patch_size': 16, 'img_size': 224}
    d = DINOiser(clip_cfg).cuda().eval()
    d.load_dino()
    d.to('cuda').eval()
    fus = SPGFusion().cuda().eval()
    sd = torch.load('CHECKPOINT/best_model_1.pth', map_location='cpu', weights_only=False)
    sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
    fus.load_state_dict(sd, strict=True)
    v = torch.randn(1, 3, 384, 512, device='cuda')
    i = torch.randn(1, 3, 384, 512, device='cuda')
    n = sum(p.numel() for p in fus.parameters()) / 1e6
    def fn():
        with torch.no_grad():
            vs, irs = d(v), d(i)
            fus(v, i, vs, irs)
    print('SPGFusion(512)  %.1f ms  params=%.2fM' % (run(fn, 5, warm=2), n))


if __name__ == '__main__':
    mode = sys.argv[1]
    (pia if mode == 'pia' else spg)()
