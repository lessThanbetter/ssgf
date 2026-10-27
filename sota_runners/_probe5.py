"""Inspect channel-importance values for the Fig.5 scenes under two protocols."""

import glob
import os

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, "/root/autodl-tmp/ssgf")
from ssgf.config import get_default_config
from ssgf.models import Encoder, ModalityClassifier
from ssgf.utils.checkpoint import load_ckpt

BASE = "/root/autodl-tmp/ssgf"
TEST = "/root/autodl-tmp/datasets/MSRS-main/test"


def prep(name, mode):
    vis = cv2.imread(f"{TEST}/vi/{name}", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    ir = cv2.imread(f"{TEST}/ir/{name}", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    if mode == "resize":
        h, w = ir.shape
        sc = 512 / max(h, w)
        nw, nh = int(w * sc), int(h * sc)
        vis, ir = cv2.resize(vis, (nw, nh)), cv2.resize(ir, (nw, nh))
    else:  # 128 center crop at native scale (training scale)
        h, w = ir.shape
        y, x = (h - 128) // 2, (w - 128) // 2
        vis, ir = vis[y:y + 128, x:x + 128], ir[y:y + 128, x:x + 128]
    return vis * 2 - 1, ir * 2 - 1


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc = Encoder(cfg)
    load_ckpt(f"{BASE}/data/msrs_csf.pt", encoder=enc)
    enc = enc.to(dev).eval()
    cls = ModalityClassifier(cfg).to(dev).eval()
    load_ckpt(f"{BASE}/data/msrs_csf.pt", classifier=cls)

    names = sorted(os.path.basename(p) for p in glob.glob(f"{TEST}/ir/*"))[:6]
    for mode in ("resize", "crop128"):
        print(f"===== protocol: {mode} =====")
        with torch.no_grad():
            for name in names:
                vis, ir = prep(name, mode)
                v = torch.from_numpy(vis)[None, None].to(dev)
                i = torch.from_numpy(ir)[None, None].to(dev)
                Fv, Fi = enc(v), enc(i)
                p0 = torch.softmax(cls(Fi), 1)[0, 1].item()
                ps = []
                for c in range(Fv.shape[1]):
                    Fr = Fi.clone(); Fr[:, c] = Fv[:, c]
                    ps.append(torch.softmax(cls(Fr), 1)[0, 1].item())
                ps = np.asarray(ps)
                print(f"  {name}: p0={p0:.3f} min={ps.min():.3f} argmin={int(ps.argmin())} "
                      f"range={ps.max()-ps.min():.3f}")


if __name__ == "__main__":
    main()
