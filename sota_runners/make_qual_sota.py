"""Build a multi-method qualitative comparison on one TNO military scene:
VIS | IR | Mean | Ours(probe) | SPGFusion | PIAFusion  (2x3 grid)."""
import argparse
import os

import cv2
import numpy as np
from PIL import Image
import torch

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, SemanticProbe, build_prior
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread


def gray(p, size):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (size[0], size[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', required=True)
    ap.add_argument('--vis_dir', default='/root/autodl-tmp/ssgf/data/tno_trial/vis')
    ap.add_argument('--ir_dir', default='/root/autodl-tmp/ssgf/data/tno_trial/ir')
    ap.add_argument('--ae', default='/root/autodl-tmp/ssgf/data/msrs_ae.pt')
    ap.add_argument('--probe', default='/root/autodl-tmp/ssgf/data/msrs_probe.pt')
    ap.add_argument('--spg', default='/root/autodl-tmp/ssgf/sota_outs/spg_tno')
    ap.add_argument('--pia', default='/root/autodl-tmp/ssgf/sota_outs/pia_tno')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae, encoder=enc, decoder=dec)
    prior = build_prior('tophat')
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt(args.probe, probe=probe_rule.probe)

    vis_path = os.path.join(args.vis_dir, args.stem + '.png')
    ir_path = os.path.join(args.ir_dir, args.stem + '.png')
    vis_c = cv2.imread(vis_path)
    vis_g = cv2.cvtColor(vis_c, cv2.COLOR_BGR2GRAY)
    ir_g = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    # common size (use spg fused dims if avail, else cap to 512)
    h, w = vis_g.shape
    scale = 512 / max(h, w)
    sz = (int(w * scale), int(h * scale))
    sz = (sz[0] - sz[0] % 8, sz[1] - sz[1] % 8)
    vis_c = cv2.resize(vis_c, sz).astype(np.float32) / 255.0
    vis_g = cv2.resize(vis_g, sz).astype(np.float32) / 255.0
    ir_g = cv2.resize(ir_g, sz).astype(np.float32) / 255.0

    v = torch.from_numpy(vis_g.astype(np.float32)).to(dev).unsqueeze(0).unsqueeze(0)
    i = torch.from_numpy(ir_g.astype(np.float32)).to(dev).unsqueeze(0).unsqueeze(0)
    f_top = top_rule.fuse_pair(v.clone(), i.clone())[0].squeeze().cpu().numpy()
    f_probe = probe_rule.fuse_pair(v.clone(), i.clone())[0].squeeze().cpu().numpy()

    spg_g = gray(os.path.join(args.spg, args.stem + '.png'), sz) / 255.0
    pia_g = gray(os.path.join(args.pia, args.stem + '.png'), sz) / 255.0

    panels = [
        (vis_c, 'VIS'), (vis_g, 'IR'), (f_top, 'Top-hat'),
        (f_probe, 'Ours (probe)'), (spg_g, 'SPGFusion'), (pia_g, 'PIAFusion'),
    ]
    pw, ph = 420, 350
    rows = []
    for r in range(2):
        row = []
        for c in range(3):
            im, lab = panels[r * 3 + c]
            im = np.asarray(im, dtype=np.float32)
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = np.clip(im, 0, 1)
            im = cv2.resize((im * 255).astype(np.uint8), (pw, ph))
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(im, (0, 0), (pw - 1, 42), (25, 25, 25), -1)
            cv2.putText(im, lab, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 3, cv2.LINE_AA)
            row.append(im)
        rows.append(np.hstack(row))
    fig = np.vstack(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, fig)
    print('saved', args.out, fig.shape)


if __name__ == '__main__':
    main()
