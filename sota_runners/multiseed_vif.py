"""Per-seed VIF of the probe rule + mean/std across seeds, on 3 datasets."""
import os
import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.fusion import SemanticSaliencyRule
from ssgf.models import Decoder, Encoder
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import find_pairs, imread
from ssgf.utils.metrics import vifp


DATASETS = [
    ('TNO', '/root/autodl-tmp/ssgf/data/tno_trial', 'vis', 'ir'),
    ('MSRS', '/root/autodl-tmp/datasets/MSRS-main/test', 'vi', 'ir'),
    ('RoadScene', '/root/autodl-tmp/datasets/RoadScene-master', 'crop_HR_visible', 'infrared'),
]
SEEDS = [1, 2, 3]


def resize01(a, max_side=512):
    h, w = a.shape
    sc = max_side / max(h, w)
    if sc >= 1:
        return a
    nw, nh = int(w * sc), int(h * sc)
    return np.asarray(Image.fromarray((a * 255).astype('uint8')).resize((nw, nh)), np.float32) / 255.0


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_default_config()
    results = {}   # (dataset, seed) -> mean VIF
    allv = {}      # dataset -> list of per-image VIF (per seed)
    for dname, root, vd, ird in DATASETS:
        pairs = find_pairs(root, vd, ird)
        for seed in SEEDS:
            enc, dec = Encoder(cfg), Decoder(cfg)
            load_ckpt(f'/root/autodl-tmp/ssgf/data/msrs_ae_seed{seed}_s.pt', encoder=enc, decoder=dec)
            rule = SemanticSaliencyRule(enc, dec, cfg, dev)
            load_ckpt(f'/root/autodl-tmp/ssgf/data/msrs_probe_seed{seed}_s.pt', probe=rule.probe)
            vals = []
            for vis_p, ir_p in pairs:
                vis = resize01(imread(vis_p))
                ir = resize01(imread(ir_p))
                # resize pair together to identical shape
                h, w = min(vis.shape[0], ir.shape[0]), min(vis.shape[1], ir.shape[1])
                sc = 512 / max(h, w)
                if sc < 1:
                    nw, nh = int(w * sc), int(h * sc)
                    vis = np.asarray(Image.fromarray((vis[:h, :w] * 255).astype('uint8')).resize((nw, nh)), np.float32) / 255.0
                    ir = np.asarray(Image.fromarray((ir[:h, :w] * 255).astype('uint8')).resize((nw, nh)), np.float32) / 255.0
                else:
                    vis, ir = vis[:h, :w], ir[:h, :w]
                v = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0).to(dev)
                i = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).to(dev)
                with torch.no_grad():
                    f = rule.fuse_pair(v, i)[0].squeeze().cpu().numpy()
                    f = resize01(f, 512)
                vals.append((vifp(vis, f) + vifp(ir, f)) / 2)
            mn = float(np.mean(vals))
            results[(dname, seed)] = mn
            allv.setdefault(dname, []).append(np.array(vals))
            print(f'{dname} seed={seed}: VIF={mn:.4f}')
    print()
    for dname, *_ in DATASETS:
        means = np.array([results[(dname, s)] for s in SEEDS])
        print(f'{dname}: mean={means.mean():.4f}  std={means.std():.4f}  '
              f'min={means.min():.4f} max={means.max():.4f}')


if __name__ == '__main__':
    main()
