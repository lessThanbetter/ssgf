"""Plot the targetness/importance maps on one TNO scene:
(a) IR, (b) top-hat targetness prior S^th, (c) learned probe map S^p, (d) fused."""
import argparse
import os

import cv2
import numpy as np
import torch

from ssgf.config import get_default_config
from ssgf.fusion import SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, build_prior
from ssgf.utils.checkpoint import load_ckpt


def gray01(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def to_u8(x):
    return np.clip(x, 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', required=True)
    ap.add_argument('--vis_dir', default='/root/autodl-tmp/ssgf/data/tno_trial/vis')
    ap.add_argument('--ir_dir', default='/root/autodl-tmp/ssgf/data/tno_trial/ir')
    ap.add_argument('--ae', default='/root/autodl-tmp/ssgf/data/msrs_ae.pt')
    ap.add_argument('--probe', default='/root/autodl-tmp/ssgf/data/msrs_probe.pt')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae, encoder=enc, decoder=dec)
    rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt(args.probe, probe=rule.probe)
    prior = build_prior('tophat')

    vis = gray01(os.path.join(args.vis_dir, args.stem + '.png'))
    ir = gray01(os.path.join(args.ir_dir, args.stem + '.png'))
    h, w = ir.shape
    scale = 512 / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    nw, nh = nw - nw % 8, nh - nh % 8
    vis = cv2.resize(vis, (nw, nh)); ir = cv2.resize(ir, (nw, nh))
    v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
    i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        s_th = prior(ir)                                   # (H,W) [0,1]
        ir_feats = enc(i)
        s_p = torch.sigmoid(rule.probe.map_logits(ir_feats))  # (1,1,H,W)
        s_p = s_p[0, 0].cpu().numpy()
        fused = rule.fuse_pair(v.clone(), i.clone())[0]     # (1,1,H,W)
        fused = fused[0, 0].cpu().numpy()

    labels = ['IR', 'Top-hat prior $S^{th}$', 'Probe map $S^{p}$', 'Fused']
    imgs = [ir, to_u8(s_th), to_u8(s_p), to_u8(fused)]
    pw, ph = 380, 380
    row = []
    for lab, im in zip(labels, imgs):
        im = cv2.resize((to_u8(im) * 255).astype(np.uint8), (pw, ph))
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(im, (0, 0), (pw - 1, 34), (25, 25, 25), -1)
        cv2.putText(im, lab, (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        row.append(im)
    fig = np.hstack(row)
    cv2.imwrite(args.out, fig)
    print('saved', args.out, fig.shape)


if __name__ == '__main__':
    main()
