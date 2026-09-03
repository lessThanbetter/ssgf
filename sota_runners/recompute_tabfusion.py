"""Recompute tab:fusion (VIF/EN/SSIM/PSNR for mean/top-hat/probe/ensemble) with the
stabilized (30-epoch) official model. Skips Qabf/SCD/MS-SSIM (slow, not in the table)."""
import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, build_prior
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import find_pairs, imread
from ssgf.utils.metrics import entropy, psnr, ssim, vifp


DATASETS = [
    ('TNO', '/root/autodl-tmp/ssgf/data/tno_trial', 'vis', 'ir'),
    ('MSRS', '/root/autodl-tmp/datasets/MSRS-main/test', 'vi', 'ir'),
    ('RoadScene', '/root/autodl-tmp/datasets/RoadScene-master', 'crop_HR_visible', 'infrared'),
]


def rz(vis, ir, max_side=512):
    h, w = vis.shape
    sc = max_side / max(h, w)
    if sc >= 1:
        return vis, ir
    nw, nh = int(w * sc), int(h * sc)
    z = lambda a: np.asarray(Image.fromarray((a * 255).astype('uint8')).resize((nw, nh)), np.float32) / 255.0
    return z(vis), z(ir)


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

    for dname, root, vd, ird in DATASETS:
        pairs = find_pairs(root, vd, ird)
        acc = {k: {m: [] for m in ['mean', 'top-hat', 'probe', 'ensemble']}
               for k in ['VIF', 'EN', 'SSIM_vis', 'SSIM_ir', 'PSNR_vis']}
        for vis_p, ir_p in pairs:
            vis, ir = rz(imread(vis_p), imread(ir_p))
            v = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0).to(dev)
            i = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).to(dev)
            with torch.no_grad():
                f_mean = mean_rule.fuse_pair(v, i)[0]
                f_top = top_rule.fuse_pair(v, i)[0]
                f_probe = probe_rule.fuse_pair(v, i)[0]
                f_ens = (f_top + f_probe) / 2
            fs = {'mean': f_mean, 'top-hat': f_top, 'probe': f_probe, 'ensemble': f_ens}
            for m, f in fs.items():
                fn = f.squeeze().cpu().numpy()
                fn = np.asarray(Image.fromarray((np.clip(fn, 0, 1) * 255).astype('uint8')).resize((vis.shape[1], vis.shape[0])), np.float32) / 255.0
                acc['VIF'][m].append((vifp(vis, fn) + vifp(ir, fn)) / 2)
                acc['EN'][m].append(entropy(fn))
                acc['SSIM_vis'][m].append(ssim(fn, vis))
                acc['SSIM_ir'][m].append(ssim(fn, ir))
                acc['PSNR_vis'][m].append(psnr(fn, vis))
        print(f'== {dname} (n={len(pairs)}) ==')
        for m in ['mean', 'top-hat', 'probe', 'ensemble']:
            row = ' & '.join(f'{np.mean(acc[k][m]):.4f}' for k in ['VIF', 'EN', 'SSIM_vis', 'SSIM_ir', 'PSNR_vis'])
            print(f'  {m:9s} {row}')


if __name__ == '__main__':
    main()
