"""Fig.4-equivalent: importance/targetness maps under different fusion rules.

Same layout as make_saliency_rules.py, but the false-colour maps are rendered
with a restrained warm "CSF" ramp (black -> amber -> cream) instead of viridis,
so they match the reference paper's orange / grayscale visual language.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from ssgf.config import get_default_config
from ssgf.fusion import SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, build_prior
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread


BASE = "D:/CSF-main/ssgf"
AE = f"{BASE}/repo/data/msrs_ae.pt"
PROBE = f"{BASE}/repo/data/msrs_probe.pt"
VIS_DIR = f"{BASE}/data/tno_demo/vis"
IR_DIR = f"{BASE}/data/tno_demo/ir"


def csf_lut():
    """Black -> deep amber -> bright orange -> cream, 256x3 uint8 ramp."""
    cmap = LinearSegmentedColormap.from_list(
        "csf_amber",
        ["#050302", "#3a1206", "#8f3d17", "#d4692f", "#f7a35a", "#ffe9c9"],
    )
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return lut


def heat(im):
    im = np.clip(im, 0, 1)
    x = cv2.normalize((im * 255).astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    return _LUT[x][:, :, ::-1]         # RGB ramp -> BGR for cv2.imwrite


def to3(im):
    im = np.clip(im, 0, 1)
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    return (im * 255).astype(np.uint8)


def main():
    stem = "Nato_camp_sequence_1800"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(AE, encoder=enc, decoder=dec)
    prior = build_prior("tophat")
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt(PROBE, probe=probe_rule.probe)

    vis = imread(f"{VIS_DIR}/{stem}.png")
    ir = imread(f"{IR_DIR}/{stem}.png")
    h, w = ir.shape
    sc = 512 / max(h, w)
    nw, nh = int(w * sc), int(h * sc)
    vis = cv2.resize(vis, (nw, nh)); ir = cv2.resize(ir, (nw, nh))
    v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
    i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        s_th = prior(ir)
        fi = enc(i)
        s_p = torch.sigmoid(probe_rule.probe.map_logits(fi))[0, 0].cpu().numpy()
        act = fi.abs().sum(1)[0].cpu().numpy()
        mean_map = np.full(ir.shape, 0.5, np.float32)
        max_map = act / (act.max() + 1e-8)
        fused = probe_rule.fuse_pair(v.clone(), i.clone())[0][0, 0].cpu().numpy()

    panels = [
        (to3(ir), "IR source (gray)"),
        (heat(s_th), "Top-hat prior"),
        (heat(s_p), "Probe map"),
        (heat(mean_map), "Mean rule (uniform)"),
        (heat(max_map), "Max rule (activity)"),
        (to3(fused), "Fused (probe)"),
    ]
    pw, ph = 300, 300
    rows = []
    for r in range(2):
        row = []
        for c in range(3):
            im, lab = panels[r * 3 + c]
            im = cv2.resize(im, (pw, ph))
            cv2.rectangle(im, (0, 0), (pw - 1, 30), (25, 25, 25), -1)
            cv2.putText(im, lab, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            row.append(im)
        rows.append(np.hstack(row))
    fig = np.vstack(rows)
    out = f"{BASE}/figures_csf/saliency_rules.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, fig)
    print("saved", out, fig.shape)


_LUT = csf_lut()


if __name__ == "__main__":
    main()
