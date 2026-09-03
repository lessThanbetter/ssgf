"""Fig.4/5-equivalent for the CSF classifier: P(infrared) per feature channel when
that IR channel is replaced by the corresponding visible channel."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 11})
import matplotlib.pyplot as plt
import numpy as np
import torch

from ssgf.config import get_default_config
from ssgf.models import Decoder, Encoder, ModalityClassifier
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_default_config()
    enc, dec = Encoder(cfg).to(dev), Decoder(cfg).to(dev)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_ae.pt', encoder=enc, decoder=dec)
    cls = ModalityClassifier(cfg).to(dev).eval()
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_csf.pt', classifier=cls)

    vis = imread('/root/autodl-tmp/ssgf/data/tno_trial/vis/Nato_camp_sequence_1800.png')
    ir = imread('/root/autodl-tmp/ssgf/data/tno_trial/ir/Nato_camp_sequence_1800.png')
    h, w = ir.shape
    sc = 512 / max(h, w)
    nw, nh = int(w * sc), int(h * sc)
    vis = np.asarray(np.asarray(vis)[:nh, :nw]); ir = np.asarray(ir)[:nh, :nw]
    v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
    i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        Fv = enc(v)  # (1,C,h,w)
        Fi = enc(i)
        C = Fv.shape[1]
        p0 = torch.softmax(cls(Fi), 1)[0, 1].item()
        ps = []
        for c in range(C):
            Fr = Fi.clone(); Fr[:, c] = Fv[:, c]
            ps.append(torch.softmax(cls(Fr), 1)[0, 1].item())
    xs = np.arange(C)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=300)
    ax.plot(xs, ps, '-o', color='#2f5aa0', ms=3, lw=1.2, label='P(IR) after replacing channel $c$')
    ax.axhline(p0, color='#c0392b', lw=1.0, ls='--', label=f'P(IR) original = {p0:.2f}')
    ax.axhline(0.5, color='#888', lw=0.8, ls=':', label='chance')
    ax.set_xlabel('feature channel index'); ax.set_ylabel('P(classified as infrared)')
    ax.set_ylim(0, 1.05); ax.set_xticks(xs[::4]); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_title('CSF classifier channel importance', fontsize=11)
    fig.tight_layout()
    fig.savefig('/root/autodl-tmp/ssgf/sota_outs/csf_channel.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig('/root/autodl-tmp/ssgf/sota_outs/csf_channel.svg', bbox_inches='tight', facecolor='white')
    fig.savefig('/root/autodl-tmp/ssgf/sota_outs/csf_channel.pdf', bbox_inches='tight', facecolor='white')
    print('saved csf_channel.{png,svg,pdf};  channels=', C, ' p0=%.3f' % p0)


if __name__ == '__main__':
    main()
