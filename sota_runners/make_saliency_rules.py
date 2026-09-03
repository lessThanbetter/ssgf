"""Fig.9-equivalent: importance/targetness maps under different fusion rules (heatmaps)."""
import argparse

import cv2
import numpy as np
import torch

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, build_prior
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread


def heat(im):
    im = np.clip(im, 0, 1)
    x = cv2.normalize((im * 255).astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(x, cv2.COLORMAP_JET)


def to3(im):
    im = np.clip(im, 0, 1)
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    return (im * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', required=True)
    ap.add_argument('--vis_dir', default='/root/autodl-tmp/ssgf/data/tno_trial/vis')
    ap.add_argument('--ir_dir', default='/root/autodl-tmp/ssgf/data/tno_trial/ir')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_ae.pt', encoder=enc, decoder=dec)
    prior = build_prior('tophat')
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_probe.pt', probe=probe_rule.probe)

    vis = imread(f'{args.vis_dir}/{args.stem}.png')
    ir = imread(f'{args.ir_dir}/{args.stem}.png')
    h, w = ir.shape
    sc = 512 / max(h, w)
    nw, nh = int(w * sc), int(h * sc)
    vis = cv2.resize(vis, (nw, nh)); ir = cv2.resize(ir, (nw, nh))
    v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
    i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        s_th = prior(ir)                                   # (H,W)
        fi = enc(i)
        s_p = torch.sigmoid(probe_rule.probe.map_logits(fi))[0, 0].cpu().numpy()
        act = fi.abs().sum(1)[0].cpu().numpy()
        # mean map = uniform, max map = activity (normalized)
        mean_map = np.full(ir.shape, 0.5, np.float32)
        max_map = act / (act.max() + 1e-8)
        fused = probe_rule.fuse_pair(v.clone(), i.clone())[0][0, 0].cpu().numpy()

    panels = [
        (to3(ir), 'IR source (gray)'),
        (heat(s_th), 'Top-hat prior'),
        (heat(s_p), 'Probe map'),
        (heat(mean_map), 'Mean rule (uniform)'),
        (heat(max_map), 'Max rule (activity)'),
        (to3(fused), 'Fused (probe)'),
    ]
    pw, ph = 300, 300
    rows = []
    for r in range(2):
        row = []
        for c in range(3):
            im, lab = panels[r * 3 + c]
            im = cv2.resize(im, (pw, ph))
            cv2.rectangle(im, (0, 0), (pw - 1, 30), (25, 25, 25), -1)
            cv2.putText(im, lab, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(im)
        rows.append(np.hstack(row))
    fig = np.vstack(rows)
    cv2.imwrite(args.out, fig)
    print('saved', args.out, fig.shape)


if __name__ == '__main__':
    main()
