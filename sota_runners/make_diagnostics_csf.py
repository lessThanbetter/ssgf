"""Three method-internal diagnostic line plots, mirroring the CSF reference.

Reference Fig.4  - channel-replacement -> P(IR): 1x3 panels, three image pairs.
Reference Fig.5  - same diagnostic, three more image pairs: 1x3 panels.
Reference Fig.6  - linear interpolation phi_i -> phi_v -> P(visible): 3 curves.

All are diagnostic curves of the method itself (not performance comparisons);
they are rendered in the reference's serif / thin-line visual language.
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
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9,
})
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from ssgf.config import get_default_config
from ssgf.models import Decoder, Encoder, ModalityClassifier
from ssgf.utils.checkpoint import load_ckpt


BASE = "D:/CSF-main/ssgf"
VIS_DIR = f"{BASE}/data/tno_trial/vis"
IR_DIR = f"{BASE}/data/tno_trial/ir"
L = 30
BLUE = "#2f5aa0"
GOLD = "#e6a23c"
YELLOW = "#d9c33b"


def load_pair(stem):
    vis = cv2.imread(f"{VIS_DIR}/{stem}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    ir = cv2.imread(f"{IR_DIR}/{stem}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    h, w = ir.shape
    sc = 512 / max(h, w)
    nw, nh = int(w * sc), int(h * sc)
    vis = cv2.resize(vis, (nw, nh)); ir = cv2.resize(ir, (nw, nh))
    # the encoder was trained on [-1,1] inputs (x*2-1)
    return vis * 2 - 1, ir * 2 - 1


def channel_importance(cls, enc, dev, pairs):
    """Return list of (xs, ps, p0) for each pair."""
    out = []
    with torch.no_grad():
        for stem in pairs:
            vis, ir = load_pair(stem)
            v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
            i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)
            Fv = enc(v); Fi = enc(i)
            C = Fv.shape[1]
            p0 = torch.softmax(cls(Fi), 1)[0, 1].item()
            ps = []
            for c in range(C):
                Fr = Fi.clone(); Fr[:, c] = Fv[:, c]
                ps.append(torch.softmax(cls(Fr), 1)[0, 1].item())
            out.append((np.arange(C), np.asarray(ps), p0, stem))
    return out


def gradient_curve(cls, enc, dev, pairs):
    """Return list of (ls, pv, stem) where pv = P(visible) over 0..L."""
    out = []
    with torch.no_grad():
        for stem in pairs:
            vis, ir = load_pair(stem)
            v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
            i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)
            Fv = enc(v); Fi = enc(i)
            pv = []
            for l in range(L + 1):
                Fl = (1 - l / L) * Fi + (l / L) * Fv
                pv.append(torch.softmax(cls(Fl), 1)[0, 0].item())
            out.append((np.arange(L + 1), np.asarray(pv), stem))
    return out


def style_axis(ax, ylim, ylab, xlab):
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_xlabel(xlab, fontsize=9)
    ax.tick_params(length=3, width=0.7, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.20, lw=0.5)


def save(fig, name):
    out = f"{BASE}/figures_csf/{name}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext, kw in {"png": dict(dpi=300), "svg": {}, "pdf": {}}.items():
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    print("saved", out, ".*")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc, dec = Encoder(cfg).to(dev), Decoder(cfg).to(dev)
    load_ckpt(f"{BASE}/repo/data/msrs_ae.pt", encoder=enc, decoder=dec)
    # IMPORTANT: the CSF classifier was trained on its OWN encoder, which differs
    # from msrs_ae.pt. Load encoder + classifier from the same checkpoint pair.
    load_ckpt(f"{BASE}/repo/data/msrs_csf.pt", encoder=enc)
    cls = ModalityClassifier(cfg).to(dev).eval()
    load_ckpt(f"{BASE}/repo/data/msrs_csf.pt", classifier=cls)

    fig4_pairs = ["Nato_camp_sequence_1800", "Duine_sequence_7400", "Tree_sequence_4901"]
    fig5_pairs = ["Nato_camp_sequence_1805", "Duine_sequence_7405", "Tree_sequence_4906"]

    # ---- Fig.4 equivalent (1x3) ----
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), dpi=300)
    for ax, (xs, ps, p0, stem) in zip(axes, channel_importance(cls, enc, dev, fig4_pairs)):
        ax.plot(xs, ps, color=BLUE, lw=1.0)
        ax.axhline(p0, color="#c0392b", lw=0.9, ls="--")
        style_axis(ax, (0, 1.05), "inferred probability", "the number of the\nreplaced feature map channel")
        ax.set_xticks(xs[::6]); ax.set_title(stem.split("_")[0], fontsize=7)
    fig.tight_layout(pad=0.6); save(fig, "diag_csf_channel_a")

    # ---- Fig.5 equivalent (1x3) ----
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), dpi=300)
    for ax, (xs, ps, p0, stem) in zip(axes, channel_importance(cls, enc, dev, fig5_pairs)):
        ax.plot(xs, ps, color=BLUE, lw=1.0)
        ax.axhline(p0, color="#c0392b", lw=0.9, ls="--")
        style_axis(ax, (0, 1.05), "inferred probability", "the number of the\nreplaced feature map channel")
        ax.set_xticks(xs[::6]); ax.set_title(stem.split("_")[0], fontsize=7)
    fig.tight_layout(pad=0.6); save(fig, "diag_csf_channel_b")

    # ---- Fig.6 equivalent (3 curves) ----
    colors = [BLUE, GOLD, YELLOW]
    fig, ax = plt.subplots(figsize=(4.2, 3.5), dpi=300)
    for (ls, pv, stem), c in zip(gradient_curve(cls, enc, dev, fig4_pairs), colors):
        ax.plot(ls, pv, color=c, lw=1.1, label=f"source image pair #{list(colors).index(c)+1}")
    style_axis(ax, (0.0, 1.0), "visible probability", "input feature maps")
    ax.set_xticks([0, L / 2, L])
    ax.set_xticklabels([r"$\varphi_i$", r"$\frac{l}{L}\varphi_i+(1-\frac{l}{L})\varphi_v$", r"$\varphi_v$"], fontsize=8)
    ax.legend(fontsize=7.5, frameon=True, edgecolor="#777", loc="center right")
    fig.tight_layout(); save(fig, "diag_csf_gradient")


if __name__ == "__main__":
    main()
