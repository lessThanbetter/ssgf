"""Quantitative comparison table (CC / SCD / Qabf / SSIM) on RoadScene, matching
the classification-saliency paper's Table I style. 8 methods: our 4 variants +
IFCNN + DenseFuse + PIAFusion + SPGFusion."""
import argparse
import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, build_prior
from ssgf.models.baselines import DenseFuse as DF, IFCNN, fuse_l1
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import find_pairs, imread
from ssgf.utils.metrics import scd, ssim, vifp


def cc(a, b):
    a = a.ravel(); b = b.ravel()
    am, bm = a.mean(), b.mean()
    va, vb = ((a - am) ** 2).mean(), ((b - bm) ** 2).mean()
    return float(((a - am) * (b - bm)).mean() / (np.sqrt(va) * np.sqrt(vb) + 1e-9))


def rz(vis, ir, max_side=512):
    h, w = vis.shape
    sc = max_side / max(h, w)
    if sc >= 1:
        return vis, ir
    nw, nh = int(w * sc), int(h * sc)
    z = lambda a: np.asarray(Image.fromarray((a * 255).astype('uint8')).resize((nw, nh)), np.float32) / 255.0
    return z(vis), z(ir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_ae.pt', encoder=enc, decoder=dec)
    prior = build_prior('tophat')
    mean_rule = MeanRule(enc, dec, cfg, dev)
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_probe.pt', probe=probe_rule.probe)
    ifcnn = IFCNN().to(dev).eval(); ifcnn.load_state_dict(torch.load('/root/autodl-tmp/ssgf/data/btl/ifcnn.pt', map_location=dev, weights_only=False))
    df = DF().to(dev).eval(); df.load_state_dict(torch.load('/root/autodl-tmp/ssgf/data/btl/densefuse.pt', map_location=dev, weights_only=False))

    root = '/root/autodl-tmp/datasets/RoadScene-master'
    pairs = find_pairs(root, 'crop_HR_visible', 'infrared')
    pia = '/root/autodl-tmp/ssgf/sota_outs/pia_roadscene'
    spg = '/root/autodl-tmp/ssgf/sota_outs/spg_roadscene'

    meth = ['mean', 'top-hat', 'probe', 'ensemble', 'IFCNN', 'DenseFuse', 'PIAFusion', 'SPGFusion']
    acc = {m: {k: [] for k in ['CC', 'SCD', 'VIF', 'SSIM']} for m in meth}

    use = pairs[:args.limit] if args.limit else pairs
    for vis_p, ir_p in use:
        name = vis_p.split('/')[-1].split('.')[0]
        vis, ir = rz(imread(vis_p), imread(ir_p))
        v = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0).to(dev)
        i = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad():
            fm = mean_rule.fuse_pair(v, i)[0].squeeze().cpu().numpy()
            ft = top_rule.fuse_pair(v, i)[0].squeeze().cpu().numpy()
            fp = probe_rule.fuse_pair(v, i)[0].squeeze().cpu().numpy()
            fe = (ft + fp) / 2
            f_if = ifcnn.decode(fuse_l1(ifcnn.encode(v * 2 - 1), ifcnn.encode(i * 2 - 1))).squeeze().cpu().numpy()
            f_if = (f_if + 1) / 2
            f_df = df.decode(fuse_l1(df.encode(v * 2 - 1), df.encode(i * 2 - 1))).squeeze().cpu().numpy()
            f_df = (f_df + 1) / 2
        f_pia = np.asarray(Image.open(f'{pia}/{name}.png').convert('L'), np.float32) / 255.0
        f_spg = np.asarray(Image.open(f'{spg}/{name}.png').convert('L'), np.float32) / 255.0
        # resize all fused to vis size
        def tov(f):
            return np.asarray(Image.fromarray((np.clip(f, 0, 1) * 255).astype('uint8')).resize((vis.shape[1], vis.shape[0])), np.float32) / 255.0
        fs = {'mean': fm, 'top-hat': ft, 'probe': fp, 'ensemble': fe,
              'IFCNN': f_if, 'DenseFuse': f_df, 'PIAFusion': f_pia, 'SPGFusion': f_spg}
        for m, f in fs.items():
            f = tov(f)
            acc[m]['CC'].append((cc(f, vis) + cc(f, ir)) / 2)
            acc[m]['SCD'].append(scd(vis, ir, f))
            acc[m]['VIF'].append((vifp(vis, f) + vifp(ir, f)) / 2)
            acc[m]['SSIM'].append((ssim(f, vis) + ssim(f, ir)) / 2)

    print('=== RoadScene quantitative comparison (CC / SCD / VIF / SSIM) ===')
    for m in meth:
        row = ' & '.join(f'{np.mean(acc[m][k]):.4f}' for k in ['CC', 'SCD', 'VIF', 'SSIM'])
        print(f'{m:10s} {row}')


if __name__ == '__main__':
    main()
