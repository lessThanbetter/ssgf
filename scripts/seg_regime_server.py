"""Regime-conditional segmentation on MSRS-test.

Second task for the regime analysis. Same fusion rules and same segmenter as the
paper's segmentation table, with the unfused visible and infrared sources added as
inputs and the visible source's RMS contrast recorded per image, so the input ranking
can be read within each contrast stratum. Per-image mIoU is saved so the split can be
recomputed; the paper's own segmentation table only kept the mean.
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/ssgf")

from ssgf.config import get_default_config  # noqa: E402
from ssgf.models import Decoder, Encoder  # noqa: E402
from ssgf.models.segnet import UNet  # noqa: E402
from ssgf.utils.checkpoint import load_ckpt  # noqa: E402
from ssgf.utils.io import find_pairs, imread  # noqa: E402
from sota_runners.k1_controls import CONTROLS, build_rules  # noqa: E402

BASE = "/root/autodl-tmp"
MSRS = f"{BASE}/datasets/MSRS-main"
AE = f"{BASE}/ssgf/data/msrs_ae.pt"
SEG = f"{BASE}/ssgf/data/segnet.pt"
OUT = f"{BASE}/ssgf/_restore/seg_regime.json"
N_CLASSES, SIZE = 9, 256


def miou(pred, gt, n):
    ious = []
    for c in range(n):
        inter = np.logical_and(pred == c, gt == c).sum()
        union = np.logical_or(pred == c, gt == c).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def eval_one(seg, img, gt, dev):
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(dev)
    x = torch.nn.functional.interpolate(x, size=(SIZE, SIZE), mode="bilinear",
                                        align_corners=False)
    with torch.no_grad():
        pred = seg(x).argmax(1)[0].cpu().numpy()
    g = np.asarray(Image.fromarray(gt.astype("uint8")).resize((SIZE, SIZE), Image.NEAREST),
                   np.int64)
    return miou(pred, g, N_CLASSES)


def main():
    dev = "cuda"
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(AE, encoder=enc, decoder=dec)
    rules = build_rules(cfg, enc, dec, dev)
    seg = UNet(in_ch=1, n_classes=N_CLASSES).to(dev)
    seg.load_state_dict(torch.load(SEG, map_location="cpu"))
    seg.eval()

    pairs = find_pairs(os.path.join(MSRS, "test"), "vi", "ir")
    lab_dir = os.path.join(MSRS, "test", "Segmentation_labels")
    print("pairs:", len(pairs), flush=True)
    out = {}
    for i, (vis_p, ir_p) in enumerate(pairs):
        stem = os.path.splitext(os.path.basename(vis_p))[0]
        lp = os.path.join(lab_dir, stem + ".png")
        if not os.path.exists(lp):
            continue
        gt = np.asarray(Image.open(lp), dtype=np.int64)
        vis, ir = imread(vis_p), imread(ir_p)
        v = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0).to(dev)
        r = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).to(dev)
        rec = {"rms_contrast": float(vis.std()), "mean_lum": float(vis.mean()), "miou": {}}
        rec["miou"]["visible source"] = eval_one(seg, vis, gt, dev)
        rec["miou"]["infrared source"] = eval_one(seg, ir, gt, dev)
        for c in CONTROLS:
            with torch.no_grad():
                f = rules[c].fuse_pair(v, r)[0]
            fu = np.clip(f.squeeze().cpu().numpy(), 0, 1)
            if fu.shape != vis.shape:
                fu = np.asarray(Image.fromarray((fu * 255).astype("uint8")).resize(
                    (vis.shape[1], vis.shape[0])), np.float32) / 255.0
            rec["miou"][c] = eval_one(seg, fu, gt, dev)
        out[stem] = rec
        if (i + 1) % 25 == 0:
            print("  %d/%d  %s" % (i + 1, len(pairs), stem), flush=True)
            json.dump(out, open(OUT, "w"), indent=2)
    json.dump(out, open(OUT, "w"), indent=2)
    print("SEG_REGIME_DONE", flush=True)


if __name__ == "__main__":
    main()
