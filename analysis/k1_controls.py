"""K1 causal-isolation controls for the targetness fusion rule.

Everything is held fixed (encoder, decoder, activity term, mx normalisation,
temperature, decoder) and only the saliency map S, or the resulting weight map,
is changed. This isolates the effect of the targetness map from the effect of
non-uniform weighting in general.

Controls
  C0 mean              w = 0.5
  C1 activity-only     S_v = S_r = 1, so d_s = A_s (classical L1-norm rule)
  C2 tophat            the paper's morphological prior (k=9, sigma=3)
  C3 probe             the paper's distilled probe
  C4 tophat-permuted   same S field, spatially permuted
  C5 probe-permuted    same S field, spatially permuted
  C6 tophat-wshuffle   C2 weights, spatially permuted (bypasses the softmax)

Per-image metric arrays are stored so that paired bootstrap statistics can be
computed afterwards by k1_stats.py.

Usage
  python -m sota_runners.k1_controls --out data/k1 --controls all
  python -m sota_runners.k1_controls --out data/k1 --control tophat --dataset MSRS
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.fusion import BaseRule, MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, build_prior
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import find_pairs, imread
from ssgf.utils.metrics import entropy, psnr, ssim, vifp


AE_CKPT = "/root/autodl-tmp/ssgf/data/msrs_ae.pt"
PROBE_CKPT = "/root/autodl-tmp/ssgf/data/msrs_probe.pt"

DATASETS = {
    "TNO": ("/root/autodl-tmp/ssgf/data/tno_trial", "vis", "ir"),
    "MSRS": ("/root/autodl-tmp/datasets/MSRS-main/test", "vi", "ir"),
    "RoadScene": ("/root/autodl-tmp/datasets/RoadScene-master", "crop_HR_visible", "infrared"),
}

CONTROLS = [
    "mean",
    "activity_only",
    "tophat",
    "probe",
    "tophat_permuted",
    "probe_permuted",
    "tophat_wshuffle",
    "probe_wshuffle",
]


def _to_np01(x: torch.Tensor) -> np.ndarray:
    """[-1,1] torch (1,1,H,W) -> (H,W) numpy in [0,1]."""
    return ((x.detach().cpu().numpy()[0, 0]) + 1.0) / 2.0


def _softmax_pair(d_vis: torch.Tensor, d_ir: torch.Tensor, temperature: float):
    """Exactly the normalisation used by the paper's rules."""
    mx = torch.max(d_vis.max(), d_ir.max()) + 1e-8
    e_v = torch.exp(d_vis / mx / temperature)
    e_i = torch.exp(d_ir / mx / temperature)
    w_v = e_v / (e_v + e_i)
    return w_v, 1.0 - w_v


def _permute(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Fixed spatial permutation over the last two dims of a (..., H, W) tensor.

    Works for the (1, H, W) maps returned by the prior path and for the
    (1, 1, H, W) maps returned by the probe and softmax paths.
    """
    g = torch.Generator(device=x.device).manual_seed(int(seed))
    n = x.shape[-1] * x.shape[-2]
    idx = torch.randperm(n, generator=g, device=x.device)
    lead = x.shape[:-2]
    flat = x.reshape(*lead, n)
    return flat[..., idx].reshape(*lead, x.shape[-2], x.shape[-1])


class ActivityOnlyRule(BaseRule):
    """C1. S is identically one, so only the feature activity drives the weights."""

    def __init__(self, encoder, decoder, cfg, device="cuda", temperature=0.3):
        super().__init__(encoder, decoder, cfg, device)
        self.temperature = temperature

    def saliency(self, vis_n, ir_n):
        a_v = self.encoder(vis_n).abs().sum(dim=1, keepdim=True)
        a_i = self.encoder(ir_n).abs().sum(dim=1, keepdim=True)
        return _softmax_pair(a_v, a_i, self.temperature)


class PermutedPriorRule(SemanticPriorRule):
    """C4. Identical prior and pipeline, S permuted in space."""

    def __init__(self, *a, perm_seed=0, **kw):
        super().__init__(*a, **kw)
        self.perm_seed = perm_seed

    def saliency(self, vis_n, ir_n):
        act_vis = self.encoder(vis_n).abs().sum(dim=1)
        act_ir = self.encoder(ir_n).abs().sum(dim=1)
        s_vis = torch.from_numpy(self.prior(_to_np01(vis_n))).to(self.device).unsqueeze(0)
        s_ir = torch.from_numpy(self.prior(_to_np01(ir_n))).to(self.device).unsqueeze(0)
        s_vis = _permute(s_vis, self.perm_seed)
        s_ir = _permute(s_ir, self.perm_seed + 1)
        d_vis, d_ir = s_vis * act_vis, s_ir * act_ir
        return _softmax_pair(d_vis, d_ir, self.temperature)


class PermutedProbeRule(SemanticSaliencyRule):
    """C5. Identical probe and pipeline, S permuted in space."""

    def __init__(self, *a, perm_seed=0, **kw):
        super().__init__(*a, **kw)
        self.perm_seed = perm_seed

    def saliency(self, vis_n, ir_n):
        vis_feats = self.encoder(vis_n)
        ir_feats = self.encoder(ir_n)
        s_vis = torch.sigmoid(self.probe.map_logits(vis_feats))
        s_ir = torch.sigmoid(self.probe.map_logits(ir_feats))
        s_vis = _permute(s_vis, self.perm_seed)
        s_ir = _permute(s_ir, self.perm_seed + 1)
        d_vis = s_vis * vis_feats.abs().sum(dim=1, keepdim=True)
        d_ir = s_ir * ir_feats.abs().sum(dim=1, keepdim=True)
        return _softmax_pair(d_vis, d_ir, self.temperature)


class WeightShuffleRule(BaseRule):
    """C6. Takes the weights of a base rule and permutes them spatially.

    The same permutation is applied to w_vis and w_ir, so the weight marginals
    and the complementarity w_vis + w_ir = 1 are both preserved and only the
    spatial alignment with the image is destroyed. This bypasses the softmax,
    so it is the strictest control of the three.
    """

    def __init__(self, encoder, decoder, cfg, base_rule, device="cuda", perm_seed=0):
        super().__init__(encoder, decoder, cfg, device)
        self.base_rule = base_rule
        self.perm_seed = perm_seed

    def saliency(self, vis_n, ir_n):
        w_v, w_i = self.base_rule.saliency(vis_n, ir_n)
        return _permute(w_v, self.perm_seed), _permute(w_i, self.perm_seed)


def rz(vis, ir, max_side=512):
    h, w = vis.shape
    sc = max_side / max(h, w)
    if sc >= 1:
        return vis, ir
    nw, nh = int(w * sc), int(h * sc)
    z = lambda a: np.asarray(
        Image.fromarray((a * 255).astype("uint8")).resize((nw, nh)), np.float32
    ) / 255.0
    return z(vis), z(ir)


def build_rules(cfg, enc, dec, dev, seed=0):
    prior = build_prior("tophat")
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    rules = {
        "mean": MeanRule(enc, dec, cfg, dev),
        "activity_only": ActivityOnlyRule(enc, dec, cfg, dev),
        "tophat": top_rule,
        "probe": probe_rule,
        "tophat_permuted": PermutedPriorRule(enc, dec, cfg, prior, dev, perm_seed=seed),
        "probe_permuted": PermutedProbeRule(enc, dec, cfg, dev, perm_seed=seed),
        "tophat_wshuffle": WeightShuffleRule(enc, dec, cfg, top_rule, dev, perm_seed=seed),
        "probe_wshuffle": WeightShuffleRule(enc, dec, cfg, probe_rule, dev, perm_seed=seed),
    }
    # Every probe-bearing rule must receive the distilled weights, otherwise the
    # permuted-probe control silently runs an untrained random head.
    for name, r in rules.items():
        if getattr(r, "probe", None) is not None:
            load_ckpt(PROBE_CKPT, probe=r.probe)
    return rules


def run(datasets, controls, out_dir, limit=None, seed=0):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(AE_CKPT, encoder=enc, decoder=dec)
    rules = build_rules(cfg, enc, dec, dev, seed=seed)

    for dname in datasets:
        root, vd, ird = DATASETS[dname]
        pairs = find_pairs(root, vd, ird)
        if limit:
            pairs = pairs[:limit]
        ddir = os.path.join(out_dir, dname)
        os.makedirs(ddir, exist_ok=True)
        print(f"[{dname}] {len(pairs)} pairs", flush=True)

        for cname in controls:
            dst = os.path.join(ddir, f"{cname}.npz")
            if os.path.exists(dst):
                print(f"  {cname}: exists, skip", flush=True)
                continue
            rule = rules[cname]
            vif, en, ssv, ssi, psv = [], [], [], [], []
            wdev = []
            for k, (vis_p, ir_p) in enumerate(pairs):
                vis, ir = rz(imread(vis_p), imread(ir_p))
                v = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0).to(dev)
                i = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).to(dev)
                with torch.no_grad():
                    f, dbg = rule.fuse_pair(v, i)
                fn = f.squeeze().cpu().numpy()
                fn = np.asarray(
                    Image.fromarray((np.clip(fn, 0, 1) * 255).astype("uint8")).resize(
                        (vis.shape[1], vis.shape[0])
                    ),
                    np.float32,
                ) / 255.0
                vif.append((vifp(vis, fn) + vifp(ir, fn)) / 2)
                en.append(entropy(fn))
                ssv.append(ssim(fn, vis))
                ssi.append(ssim(fn, ir))
                psv.append(psnr(fn, vis))
                w = dbg["w_vis"].float()
                wdev.append(float((w - 0.5).abs().mean()))
                if (k + 1) % 25 == 0:
                    print(
                        f"  {cname}: {k+1}/{len(pairs)} VIF={np.mean(vif):.4f}",
                        flush=True,
                    )
            np.savez(
                dst,
                vif=np.array(vif),
                en=np.array(en),
                ssim_vis=np.array(ssv),
                ssim_ir=np.array(ssi),
                psnr_vis=np.array(psv),
                w_dev=np.array(wdev),
                pairs=np.array([p[0] for p in pairs]),
            )
            print(
                f"  {cname}: done VIF={np.mean(vif):.4f} EN={np.mean(en):.4f} "
                f"w_dev={np.mean(wdev):.4f} -> {dst}",
                flush=True,
            )

        summ = {}
        for cname in controls:
            p = os.path.join(ddir, f"{cname}.npz")
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                summ[cname] = {
                    "VIF": float(z["vif"].mean()),
                    "EN": float(z["en"].mean()),
                    "SSIM_vis": float(z["ssim_vis"].mean()),
                    "SSIM_ir": float(z["ssim_ir"].mean()),
                    "PSNR_vis": float(z["psnr_vis"].mean()),
                    "w_dev": float(z["w_dev"].mean()),
                    "n": int(z["vif"].size),
                }
        with open(os.path.join(ddir, "summary.json"), "w") as fh:
            json.dump(summ, fh, indent=2)
        print(f"[{dname}] summary -> {os.path.join(ddir, 'summary.json')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/ssgf/data/k1")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--control", default="all")
    ap.add_argument("--controls", default=None, help="comma list, overrides --control")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    datasets = list(DATASETS) if a.datasets == "all" else a.datasets.split(",")
    if a.controls:
        controls = a.controls.split(",")
    elif a.control == "all":
        controls = list(CONTROLS)
    else:
        controls = a.control.split(",")
    os.makedirs(a.out, exist_ok=True)
    run(datasets, controls, a.out, limit=a.limit, seed=a.seed)


if __name__ == "__main__":
    main()
