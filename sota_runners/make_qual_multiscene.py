"""Fig.8-equivalent: multi-scene (3) x multi-method (6) qualitative with per-panel
CC/SCD/VIF/SSIM values and a red ROI zoom inset per row."""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import imread
from ssgf.utils.metrics import scd, ssim, vifp
from ssgf.utils.io import find_pairs


def cc(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    am, bm = a.mean(), b.mean()
    va = np.mean((a - am) ** 2); vb = np.mean((b - bm) ** 2)
    return float(np.mean((a - am) * (b - bm)) / (np.sqrt(va) * np.sqrt(vb) + 1e-9))


def rz(vis, ir, max_side=512):
    h, w = vis.shape
    sc = max_side / max(h, w)
    if sc >= 1:
        return vis, ir, (w, h)
    nw, nh = int(w * sc), int(h * sc)
    z = lambda a: np.asarray(Image.fromarray((a * 255).astype('uint8')).resize((nw, nh)), np.float32) / 255.0
    return z(vis), z(ir), (nw, nh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_ae.pt', encoder=enc, decoder=dec)
    mean_rule = MeanRule(enc, dec, cfg, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_probe.pt', probe=probe_rule.probe)

    # 3 scenes across datasets
    dses = [
        ('TNO', '/root/autodl-tmp/ssgf/data/tno_trial', 'vis', 'ir', 'Nato_camp_sequence_1800'),
        ('RoadScene', '/root/autodl-tmp/datasets/RoadScene-master', 'crop_HR_visible', 'infrared', None),
        ('MSRS', '/root/autodl-tmp/datasets/MSRS-main/test', 'vi', 'ir', None),
    ]
    # resolve None stems from dir list
    scenes = []
    for name, root, vd, ird, stem in dses:
        fuse_dir = {'TNO': 'tno', 'RoadScene': 'roadscene', 'MSRS': 'msrs'}[name]
        pairs = find_pairs(root, vd, ird)
        spgd = f'/root/autodl-tmp/ssgf/sota_outs/spg_{fuse_dir}'
        chosen = None
        for vp, ip in pairs:
            s = os.path.splitext(os.path.basename(vp))[0]
            if os.path.exists(f'{spgd}/{s}.png'):
                chosen = (vp, ip, s)
                break
        if chosen is None:
            chosen = (pairs[0][0], pairs[0][1], os.path.splitext(os.path.basename(pairs[0][0]))[0])
        scenes.append((name, chosen[0], chosen[1], chosen[2], fuse_dir))

    pw, ph = 300, 300
    rows = []
    for name, vis_path, ir_path, stem, fuse_dir in scenes:
        vis = imread(vis_path)
        ir = imread(ir_path)
        vis, ir, (nw, nh) = rz(vis, ir)
        v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
        i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            fm = mean_rule.fuse_pair(v, i)[0][0, 0].cpu().numpy()
            fp = probe_rule.fuse_pair(v, i)[0][0, 0].cpu().numpy()
        spg = np.asarray(Image.open(f'/root/autodl-tmp/ssgf/sota_outs/spg_{fuse_dir}/{stem}.png').convert('L'), np.float32) / 255.0
        pia = np.asarray(Image.open(f'/root/autodl-tmp/ssgf/sota_outs/pia_{fuse_dir}/{stem}.png').convert('L'), np.float32) / 255.0
        def tov(f):
            return np.asarray(Image.fromarray((np.clip(f, 0, 1) * 255).astype('uint8')).resize((nw, nh)), np.float32) / 255.0
        fs = {'mean': tov(fm), 'probe': tov(fp), 'SPGFusion': tov(spg), 'PIAFusion': tov(pia)}
        panels = [('VIS', vis), ('IR', ir)]
        for m, f in fs.items():
            mval = (cc(f, vis) + cc(f, ir)) / 2
            scd_v = scd(vis, ir, f)
            vif = (vifp(vis, f) + vifp(ir, f)) / 2
            sm = (ssim(f, vis) + ssim(f, ir)) / 2
            panels.append((m, f, f'CC {mval:.2f} SCD {scd_v:.2f} VIF {vif:.2f} SSIM {sm:.2f}'))
        row = []
        for idx, (lab, im, *met) in enumerate(panels):
            im = tov(im)
            im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
            im = cv2.resize(im, (pw, ph))
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            if lab == 'IR':
                # red ROI box around the building/target (center-right)
                cv2.rectangle(im, (int(pw * 0.55), int(ph * 0.45)), (pw - 20, ph - 40), (0, 0, 255), 2)
            cv2.rectangle(im, (0, 0), (pw - 1, 34), (20, 20, 20), -1)
            label = f'{name}: {lab}'
            cv2.putText(im, label[:20], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            if met:
                cv2.rectangle(im, (0, ph - 52), (pw - 1, ph - 1), (0, 0, 0), -1)
                cv2.putText(im, met[0], (4, ph - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            row.append(im)
        rows.append(np.hstack(row))
    fig = np.vstack(rows)
    cv2.imwrite(args.out, fig)
    print('saved', args.out, fig.shape)


if __name__ == '__main__':
    main()
