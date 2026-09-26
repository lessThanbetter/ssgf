"""Shared helpers for the evaluator-robust experiments."""
import os
import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/ssgf")

BASE = "/root/autodl-tmp"
FLIR = f"{BASE}/datasets/flir"
OUT = f"{BASE}/ssgf/sota_outs"
RES = f"{BASE}/ssgf/data/j10_robust"

MODELS = {
    "robrr": f"{BASE}/ssgf/data/msrs_robrr.pt",
    "robwc": f"{BASE}/ssgf/data/msrs_robwc.pt",
    "proberr": f"{BASE}/ssgf/data/msrs_proberr.pt",
    "probeone": f"{BASE}/ssgf/data/msrs_probeone.pt",
}


def build(ckpt):
    from ssgf.config import get_default_config
    from ssgf.models import Decoder, Encoder, build_probe
    from ssgf.utils.checkpoint import load_ckpt

    cfg = get_default_config()
    cfg.decoder_tanh = True
    enc, dec = Encoder(cfg), Decoder(cfg)
    probe = build_probe(cfg)
    load_ckpt(ckpt, encoder=enc, decoder=dec, probe=probe)
    enc = enc.cuda().eval()
    dec = dec.cuda().eval()
    probe = probe.cuda().eval()
    for m in (enc, dec, probe):
        for p in m.parameters():
            p.requires_grad_(False)
    return cfg, enc, dec, probe


def fuse_pair(cfg, enc, dec, probe, vis, ir, tau=0.3):
    import numpy as np

    v = torch.from_numpy(vis)[None, None].cuda() * 2 - 1
    r = torch.from_numpy(ir)[None, None].cuda() * 2 - 1
    with torch.no_grad():
        fv, fi = enc(v), enc(r)
        sv = torch.sigmoid(probe.map_logits(fv))
        si = torch.sigmoid(probe.map_logits(fi))
        dv = sv * fv.abs().sum(1, keepdim=True)
        di = si * fi.abs().sum(1, keepdim=True)
        mx = torch.max(dv.max(), di.max()) + 1e-8
        ev = torch.exp(dv / mx / tau)
        ei = torch.exp(di / mx / tau)
        wv = ev / (ev + ei)
        f = dec(wv * fv + (1 - wv) * fi)
    return ((f + 1) / 2)[0, 0].clamp(0, 1).cpu().numpy()
