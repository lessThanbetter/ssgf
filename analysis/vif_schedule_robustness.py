"""Ranking robustness of the fusion VIF to the Gaussian schedule of the index.

The released implementation builds the multiscale filter with sigma = 2^(scale-1)
for windows of 17, 9, 5 and 3 pixels. The canonical single-reference pixel VIF
uses sigma = window/5, namely 3.4, 1.8, 1.0 and 0.6 for the same windows. This
script fuses the LLVIP test pairs with the released checkpoints at the paper's
512 px evaluation scale and scores every fused output under both schedules.
"""

import json
import os
import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

BASE = r"D:\CSF-main\ssgf"
sys.path.insert(0, BASE)

import torch  # noqa: E402

from ssgf.config import get_default_config  # noqa: E402
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule  # noqa: E402
from ssgf.models import Decoder, Encoder, build_prior  # noqa: E402
from ssgf.utils.checkpoint import load_ckpt  # noqa: E402
from ssgf.utils.metrics import _gauss_kernel, vifp  # noqa: E402

TEST = os.path.join(BASE, "data", "det_subset", "test")
AE = os.path.join(BASE, "repo", "data", "msrs_ae.pt")
PROBE = os.path.join(BASE, "repo", "data", "msrs_probe.pt")
LIMIT = 30
MAX_SIDE = 512


def imread(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return a


def resize_pair(a, b, max_side):
    h, w = a.shape
    scale = max_side / max(h, w)
    if scale >= 1:
        return a, b
    nw, nh = int(w * scale), int(h * scale)

    def rz(x):
        return np.asarray(Image.fromarray((x * 255).astype("uint8")).resize((nw, nh)),
                          dtype=np.float32) / 255.0

    return rz(a), rz(b)


def vifp_canonical(ref, dist, sigma_nsq=2.0):
    """Single-reference pixel VIF with the canonical sigma = window / 5."""
    num, den = 0.0, 0.0
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        g = _gauss_kernel(n, n / 5.0)
        if scale > 1:
            ref = ndi.convolve(ref, g, mode="nearest")
            dist = ndi.convolve(dist, g, mode="nearest")
        mu1 = ndi.convolve(ref, g, mode="nearest")
        mu2 = ndi.convolve(dist, g, mode="nearest")
        s11 = ndi.convolve(ref * ref, g, mode="nearest") - mu1 * mu1
        s22 = ndi.convolve(dist * dist, g, mode="nearest") - mu2 * mu2
        s12 = ndi.convolve(ref * dist, g, mode="nearest") - mu1 * mu2
        g_map = s12 / (s11 + 1e-9)
        sv_sq = s22 - g_map * s12
        g_map = np.clip(g_map, 0, 1)
        sv_sq[sv_sq < 0] = 0
        num += np.sum(np.log10(1 + g_map ** 2 * s11 / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + s11 / sigma_nsq))
    return float(num / (den + 1e-9))


def vif_pair(f, vis, ir, fn):
    return 0.5 * (fn(vis, f) + fn(ir, f))


def main():
    os.makedirs(os.path.join(BASE, "_review_ipt", "_vifrob"), exist_ok=True)
    cfg = get_default_config(probe_type="single", backbone="csf")
    device = "cpu"
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(AE, encoder=enc, decoder=dec)
    prior = build_prior("tophat")
    mean_rule = MeanRule(enc, dec, cfg, device)
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, device)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, device)
    load_ckpt(PROBE, probe=probe_rule.probe)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(TEST, "visible")))
    stems = stems[:LIMIT]

    rows = {m: {"rel": [], "can": []} for m in ("mean", "tophat", "probe", "ensemble")}

    def fuse(rule, vis, ir):
        v = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0)
        i = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0)
        f, _ = rule.fuse_pair(v, i)
        return f.squeeze().detach().cpu().numpy()

    for idx, stem in enumerate(stems):
        vis = imread(os.path.join(TEST, "visible", stem + ".jpg"))
        ir = imread(os.path.join(TEST, "infrared", stem + ".jpg"))
        vis, ir = resize_pair(vis, ir, MAX_SIDE)
        outs = {
            "mean": fuse(mean_rule, vis, ir),
            "tophat": fuse(top_rule, vis, ir),
            "probe": fuse(probe_rule, vis, ir),
        }
        outs["ensemble"] = 0.5 * (outs["tophat"] + outs["probe"])
        for m, f in outs.items():
            rows[m]["rel"].append(vif_pair(f, vis, ir, vifp))
            rows[m]["can"].append(vif_pair(f, vis, ir, vifp_canonical))
        print("[vifrob] %d/%d %s" % (idx + 1, len(stems), stem), flush=True)

    out = {"limit": LIMIT, "max_side": MAX_SIDE, "methods": {}}
    rel_means, can_means = {}, {}
    for m, d in rows.items():
        rel_means[m] = float(np.mean(d["rel"]))
        can_means[m] = float(np.mean(d["can"]))
        out["methods"][m] = {
            "vif_released_schedule": rel_means[m],
            "vif_canonical_schedule": can_means[m],
            "per_image_released": [round(v, 4) for v in d["rel"]],
            "per_image_canonical": [round(v, 4) for v in d["can"]],
        }

    order_rel = sorted(rel_means, key=lambda k: -rel_means[k])
    order_can = sorted(can_means, key=lambda k: -can_means[k])
    out["ranking_released"] = order_rel
    out["ranking_canonical"] = order_can
    out["ranking_identical"] = order_rel == order_can
    out["max_abs_difference"] = float(max(abs(rel_means[m] - can_means[m]) for m in rel_means))
    out["mean_abs_difference"] = float(np.mean([abs(rel_means[m] - can_means[m]) for m in rel_means]))
    # per-image agreement: how often the two schedules order two variants the same way
    agree = tot = 0
    ms = list(rows)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = rows[ms[i]], rows[ms[j]]
            for x, y in zip(a["rel"], b["rel"]):
                for x2, y2 in zip(a["can"], b["can"]):
                    pass
            for k in range(len(a["rel"])):
                if (a["rel"][k] - b["rel"][k]) * (a["can"][k] - b["can"][k]) > 0:
                    agree += 1
                tot += 1
    out["pairwise_order_agreement"] = agree / tot if tot else None

    with open(os.path.join(BASE, "_review_ipt", "_vifrob", "schedule_robustness.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "methods"}, indent=2))
    print("released:", json.dumps(rel_means, indent=2))
    print("canonical:", json.dumps(can_means, indent=2))
    print("VIFROB_DONE", flush=True)


if __name__ == "__main__":
    main()
