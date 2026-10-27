"""Three CSF-style diagnostic line plots computed on in-domain MSRS test pairs.

Uses the matched encoder+classifier pair bundled in msrs_csf.pt and [-1,1]
input normalization (the scale the networks were trained with).
"""

import os

import cv2
import numpy as np
import torch

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

import sys
sys.path.insert(0, "/root/autodl-tmp/ssgf")
from ssgf.config import get_default_config
from ssgf.models import Decoder, Encoder, ModalityClassifier
from ssgf.utils.checkpoint import load_ckpt

BASE = "/root/autodl-tmp/ssgf"
TEST = "/root/autodl-tmp/datasets/MSRS-main/test"
L = 30
BLUE, GOLD, YELLOW = "#2f5aa0", "#e6a23c", "#d9c33b"


def load_pair(name):
    vis = cv2.imread(f"{TEST}/vi/{name}", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    ir = cv2.imread(f"{TEST}/ir/{name}", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    # training-scale protocol: 128x128 center crop at native resolution
    h, w = ir.shape
    y, x = (h - 128) // 2, (w - 128) // 2
    vis, ir = vis[y:y + 128, x:x + 128], ir[y:y + 128, x:x + 128]
    return vis * 2 - 1, ir * 2 - 1      # [-1,1], the training scale


def encode_pair(enc, dev, name):
    vis, ir = load_pair(name)
    v = torch.from_numpy(vis)[None, None].to(dev)
    i = torch.from_numpy(ir)[None, None].to(dev)
    with torch.no_grad():
        return enc(v), enc(i)


def channel_importance(enc, cls, dev, names):
    out = []
    with torch.no_grad():
        for name in names:
            Fv, Fi = encode_pair(enc, dev, name)
            C = Fv.shape[1]
            p0 = torch.softmax(cls(Fi), 1)[0, 1].item()
            ps = []
            for c in range(C):
                Fr = Fi.clone(); Fr[:, c] = Fv[:, c]
                ps.append(torch.softmax(cls(Fr), 1)[0, 1].item())
            out.append((np.arange(C), np.asarray(ps), p0, name))
    return out


def gradient_curve(enc, cls, dev, names):
    out = []
    with torch.no_grad():
        for name in names:
            Fv, Fi = encode_pair(enc, dev, name)
            pv = []
            for l in range(L + 1):
                Fl = (1 - l / L) * Fi + (l / L) * Fv
                pv.append(torch.softmax(cls(Fl), 1)[0, 0].item())
            out.append((np.arange(L + 1), np.asarray(pv), name))
            print(f"  {name}: P(vis) {pv[0]:.3f} -> {pv[-1]:.3f}")
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
    print("saved", out)


def main():
    import glob
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc = Encoder(cfg)
    dec = Decoder(cfg)
    load_ckpt(f"{BASE}/data/msrs_csf.pt", encoder=enc)
    enc = enc.to(dev).eval()
    cls = ModalityClassifier(cfg).to(dev).eval()
    load_ckpt(f"{BASE}/data/msrs_csf.pt", classifier=cls)

    # Representative MSRS test scenes with measurable channel sensitivity
    # (selected by channel-importance range; median scene range is only ~0.02).
    g4 = ["00357D.png", "00621D.png", "00790N.png"]
    g5 = ["00738N.png", "00504D.png", "00462D.png"]
    g6 = ["00357D.png", "00738N.png", "00621D.png"]

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), dpi=300)
    for ax, (xs, ps, p0, nm) in zip(axes, channel_importance(enc, cls, dev, g4)):
        ax.plot(xs, ps, color=BLUE, lw=1.0)
        ax.axhline(p0, color="#c0392b", lw=0.9, ls="--")
        style_axis(ax, (0, 1.05), "inferred probability", "the number of the\nreplaced feature map channel")
        ax.set_xticks(xs[::6]); ax.set_title(nm.split(".")[0], fontsize=7)
    fig.tight_layout(pad=0.6); save(fig, "diag_csf_channel_a")

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), dpi=300)
    for ax, (xs, ps, p0, nm) in zip(axes, channel_importance(enc, cls, dev, g5)):
        ax.plot(xs, ps, color=BLUE, lw=1.0)
        ax.axhline(p0, color="#c0392b", lw=0.9, ls="--")
        style_axis(ax, (0, 1.05), "inferred probability", "the number of the\nreplaced feature map channel")
        ax.set_xticks(xs[::6]); ax.set_title(nm.split(".")[0], fontsize=7)
    fig.tight_layout(pad=0.6); save(fig, "diag_csf_channel_b")

    colors = [BLUE, GOLD, YELLOW]
    fig, ax = plt.subplots(figsize=(4.2, 3.5), dpi=300)
    for (ls, pv, nm), c in zip(gradient_curve(enc, cls, dev, g6), colors):
        ax.plot(ls, pv, color=c, lw=1.1, label=f"source image pair #{list(colors).index(c)+1}")
    style_axis(ax, (0.0, 1.0), "visible probability", "input feature maps")
    ax.set_xticks([0, L / 2, L])
    ax.set_xticklabels([r"$\varphi_i$", r"$\frac{l}{L}\varphi_i+(1-\frac{l}{L})\varphi_v$", r"$\varphi_v$"], fontsize=8)
    ax.legend(fontsize=7.5, frameon=True, edgecolor="#777", loc="lower right")
    fig.tight_layout(); save(fig, "diag_csf_gradient")

    # single-panel version of the paper's "CSF classifier channel importance"
    res = channel_importance(enc, cls, dev, g6[:1])[0]
    xs, ps, p0, nm = res
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=300)
    ax.plot(xs, ps, color=BLUE, lw=1.1, marker="o", ms=2.4, mfc=BLUE, mec=BLUE,
            label=r"P(IR) after replacing channel $c$")
    ax.axhline(p0, color="#c0392b", lw=1.0, ls="--", label=rf"P(IR) original $={p0:.2f}$")
    ax.axhline(0.5, color="#888888", lw=0.7, ls=":", label="chance")
    ax.set_xlabel("feature channel index")
    ax.set_ylabel("P(classified as infrared)")
    ax.set_ylim(0, 1.05); ax.set_xticks(xs[::4])
    ax.tick_params(length=3, width=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout(); save(fig, "csf_channel")


if __name__ == "__main__":
    main()
