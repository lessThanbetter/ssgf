"""CSF classifier channel-importance line plot, restyled to reference Fig.4/5.

Thin blue line, serif (Times-like) labels, no grid, minimal markers -- the
visual language of Xu et al. (IEEE TCI 2021) Fig. 4 / Fig. 5.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 11,
})
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from ssgf.config import get_default_config
from ssgf.models import Decoder, Encoder, ModalityClassifier
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread


BASE = "D:/CSF-main/ssgf"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc, dec = Encoder(cfg).to(dev), Decoder(cfg).to(dev)
    load_ckpt(f"{BASE}/repo/data/msrs_ae.pt", encoder=enc, decoder=dec)
    cls = ModalityClassifier(cfg).to(dev).eval()
    load_ckpt(f"{BASE}/repo/data/msrs_csf.pt", classifier=cls)

    stem = "Nato_camp_sequence_1800"
    vis = imread(f"{BASE}/data/tno_demo/vis/{stem}.png")
    ir = imread(f"{BASE}/data/tno_demo/ir/{stem}.png")
    h, w = ir.shape
    sc = 512 / max(h, w)
    nw, nh = int(w * sc), int(h * sc)
    vis = np.asarray(vis)[:nh, :nw]; ir = np.asarray(ir)[:nh, :nw]
    v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
    i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        Fv = enc(v); Fi = enc(i)
        C = Fv.shape[1]
        p0 = torch.softmax(cls(Fi), 1)[0, 1].item()
        ps = []
        for c in range(C):
            Fr = Fi.clone(); Fr[:, c] = Fv[:, c]
            ps.append(torch.softmax(cls(Fr), 1)[0, 1].item())

    xs = np.arange(C)
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=300)
    ax.plot(xs, ps, color="#2f5aa0", lw=1.1, ms=2.4, marker="o", mfc="#2f5aa0",
            mec="#2f5aa0", label=r"P(IR) after replacing channel $c$")
    ax.axhline(p0, color="#c0392b", lw=1.0, ls="--",
               label=rf"P(IR) original $={p0:.2f}$")
    ax.axhline(0.5, color="#888888", lw=0.7, ls=":", label="chance")
    ax.set_xlabel("feature channel index")
    ax.set_ylabel("P(classified as infrared)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs[::4])
    ax.tick_params(length=3, width=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()

    out = f"{BASE}/figures_csf/csf_channel"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext, kw in {"png": dict(dpi=300), "svg": {}, "pdf": {}}.items():
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    print(f"saved {out}.{{png,svg,pdf}}  channels={C}  p0={p0:.3f}")


if __name__ == "__main__":
    main()
