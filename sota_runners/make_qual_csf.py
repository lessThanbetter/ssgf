"""CSF-style qualitative plate (landscape): rows = scenes, columns = methods.

Each fused panel carries a clean light strip with CC/SCD/VIF/SSIM below it; the IR
panel marks the target ROI with a red box; the last column shows the zoomed ROI of
the probe fusion.  The plate is landscape (approx 2.4:1) so that it reads well at
page width instead of filling a whole page vertically.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, build_prior
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread, find_pairs
from ssgf.utils.metrics import scd, ssim, vifp


def cc(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    am, bm = a.mean(), b.mean()
    va = np.mean((a - am) ** 2)
    vb = np.mean((b - bm) ** 2)
    return float(np.mean((a - am) * (b - bm)) / (np.sqrt(va) * np.sqrt(vb) + 1e-9))


def rz(vis, ir, max_side=512):
    h, w = vis.shape
    sc = max_side / max(h, w)
    if sc >= 1:
        return vis, ir, (w, h)
    nw, nh = int(w * sc), int(h * sc)
    z = lambda a: np.asarray(
        Image.fromarray((a * 255).astype("uint8")).resize((nw, nh)), np.float32
    ) / 255.0
    return z(vis), z(ir), (nw, nh)


def tov(f, shape):
    return np.asarray(
        Image.fromarray((np.clip(f, 0, 1) * 255).astype("uint8")).resize(
            (shape[0], shape[1])
        ),
        np.float32,
    ) / 255.0


def gray_bgr(im):
    return cv2.cvtColor((np.clip(im, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def center_text(img, text, cx, cy, scale, color, thick=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, text, (int(cx - tw / 2), int(cy + th / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def bar(h, w, bg):
    return np.full((h, w, 3), bg, np.uint8)


def px_scale(px, thick=2, ref="Scene"):
    """fontScale so the rendered pixel height of `ref` equals `px`
    (calibrated by actually rasterising the reference string)."""
    tmp = np.zeros((160, 600), np.uint8)
    cv2.putText(tmp, ref, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, thick,
                cv2.LINE_AA)
    ys, _ = np.where(tmp > 0)
    return px / float(ys.max() - ys.min() + 1)


def fit_scale(text, max_w, scale0=0.55, smin=0.26):
    s = scale0
    while s > smin:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
        if tw <= max_w:
            return s
        s -= 0.02
    return smin


def _num(v):
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0") else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=340)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt("/root/autodl-tmp/ssgf/data/msrs_ae.pt", encoder=enc, decoder=dec)
    prior = build_prior("tophat")
    mean_rule = MeanRule(enc, dec, cfg, dev)
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt("/root/autodl-tmp/ssgf/data/msrs_probe.pt", probe=probe_rule.probe)

    # scene spec and a chosen stem with an available SPGFusion output
    scenes = [
        ("TNO", "/root/autodl-tmp/ssgf/data/tno_trial", "vis", "ir", "tno"),
        ("RoadScene", "/root/autodl-tmp/datasets/RoadScene-master",
         "crop_HR_visible", "infrared", "roadscene"),
        ("MSRS", "/root/autodl-tmp/datasets/MSRS-main/test", "vi", "ir", "msrs"),
    ]
    data = []  # per scene: dict of fused arrays (float32 [0,1] at vis shape)
    for name, root, vd, ird, fd in scenes:
        pairs = find_pairs(root, vd, ird)
        stem = None
        for vp, ip in pairs:
            s = os.path.splitext(os.path.basename(vp))[0]
            if os.path.exists(f"/root/autodl-tmp/ssgf/sota_outs/spg_{fd}/{s}.png"):
                stem = s
                vis_p, ir_p = vp, ip
                break
        if stem is None:
            vis_p, ir_p = pairs[0]
            stem = os.path.splitext(os.path.basename(vis_p))[0]
        vis = imread(vis_p)
        ir = imread(ir_p)
        vis, ir, (nw, nh) = rz(vis, ir)
        v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
        i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            fm = mean_rule.fuse_pair(v, i)[0][0, 0].cpu().numpy()
            ft = top_rule.fuse_pair(v, i)[0][0, 0].cpu().numpy()
            fp = probe_rule.fuse_pair(v, i)[0][0, 0].cpu().numpy()
        spg = np.asarray(Image.open(
            f"/root/autodl-tmp/ssgf/sota_outs/spg_{fd}/{stem}.png").convert("L"),
            np.float32) / 255.0
        pia = np.asarray(Image.open(
            f"/root/autodl-tmp/ssgf/sota_outs/pia_{fd}/{stem}.png").convert("L"),
            np.float32) / 255.0
        fused = {
            "Mean": tov(fm, (nw, nh)),
            "Top-hat": tov(ft, (nw, nh)),
            "Probe": tov(fp, (nw, nh)),
            "SPGFusion": tov(spg, (nw, nh)),
            "PIAFusion": tov(pia, (nw, nh)),
        }
        data.append((name, vis, ir, fused, (nw, nh)))

    methods = ["VIS", "IR", "Mean", "Top-hat", "Probe", "SPGFusion", "PIAFusion"]
    c = args.cell
    strip = max(30, c // 9)           # light metric strip height
    tile = c + strip
    label_w = int(c * 0.62)
    head_h = max(52, c // 6)
    cols = methods + ["ROI zoom"]
    total_w = label_w + len(cols) * c

    lab_px, lab_th = 24, 3            # left + top label height (px) and stroke width
    lab_sc = px_scale(lab_px, lab_th)
    head = bar(head_h, total_w, 45)
    center_text(head, "Scene", label_w // 2, head_h // 2, lab_sc, (255, 255, 255), lab_th)
    for j, m in enumerate(cols):
        center_text(head, m, label_w + j * c + c // 2, head_h // 2,
                    lab_sc, (255, 255, 255), lab_th)

    rows = [head]
    for sname, vis, ir, fused, (nw, nh) in data:
        lab = bar(tile, label_w, 60)
        center_text(lab, sname, label_w // 2, tile // 2, lab_sc, (255, 255, 255), lab_th)
        cells = [lab]
        for m in methods:
            if m == "VIS":
                im = gray_bgr(vis)
            elif m == "IR":
                im = gray_bgr(ir)
            else:
                im = gray_bgr(fused[m])
            im = cv2.resize(im, (c, c))
            # mark the same target ROI on every panel
            cv2.rectangle(im, (int(c * 0.5), int(c * 0.38)),
                          (int(c * 0.98), int(c * 0.92)), (0, 0, 255), 3)
            cv2.rectangle(im, (0, 0), (c - 1, c - 1), (0, 0, 0), 1)
            st = bar(strip, c, 238)
            if m not in ("VIS", "IR"):
                f = fused[m]
                mm = (cc(f, vis) + cc(f, ir)) / 2
                sd = scd(vis, ir, f)
                vf = (vifp(vis, f) + vifp(ir, f)) / 2
                sm = (ssim(f, vis) + ssim(f, ir)) / 2
                txt = (f"CC {_num(mm)}  SCD {_num(sd)}  "
                       f"VIF {_num(vf)}  SSIM {_num(sm)}")
                center_text(st, txt, c // 2, strip // 2,
                            fit_scale(txt, c - 12), (25, 25, 25), 1)
            cv2.rectangle(st, (0, 0), (c - 1, strip - 1), (0, 0, 0), 1)
            cells.append(np.vstack([im, st]))
        # ROI zoom of the probe fusion, aligned as the last column
        x0, y0 = int(nw * 0.5), int(nh * 0.38)
        x1, y1 = int(nw * 0.98), int(nh * 0.92)
        crop = np.asarray(Image.fromarray(
            (np.clip(fused["Probe"], 0, 1) * 255).astype("uint8")).crop((x0, y0, x1, y1)))
        crop = cv2.resize(crop, (c, c), interpolation=cv2.INTER_CUBIC)
        z = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(z, (0, 0), (c - 1, c - 1), (0, 0, 0), 1)
        zst = bar(strip, c, 238)
        cv2.rectangle(zst, (0, 0), (c - 1, strip - 1), (0, 0, 0), 1)
        cells.append(np.vstack([z, zst]))
        rows.append(np.hstack(cells))

    fig = np.vstack(rows)
    cv2.imwrite(args.out, fig)
    print("saved", args.out, fig.shape)


if __name__ == "__main__":
    main()
