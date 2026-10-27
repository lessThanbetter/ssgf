"""Rank MSRS test scenes by channel-importance range (training-scale protocol)."""

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


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc = Encoder(cfg)
    load_ckpt(f"{BASE}/data/msrs_csf.pt", encoder=enc)
    enc = enc.to(dev).eval()
    cls = ModalityClassifier(cfg).to(dev).eval()
    load_ckpt(f"{BASE}/data/msrs_csf.pt", classifier=cls)

    names = sorted(os.path.basename(p) for p in glob.glob(f"{TEST}/ir/*"))
    rows = []
    with torch.no_grad():
        for name in names:
            vis = cv2.imread(f"{TEST}/vi/{name}", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            ir = cv2.imread(f"{TEST}/ir/{name}", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            h, w = ir.shape
            y, x = (h - 128) // 2, (w - 128) // 2
            vis, ir = vis[y:y + 128, x:x + 128] * 2 - 1, ir[y:y + 128, x:x + 128] * 2 - 1
            v = torch.from_numpy(vis)[None, None].to(dev)
            i = torch.from_numpy(ir)[None, None].to(dev)
            Fv, Fi = enc(v), enc(i)
            p0 = torch.softmax(cls(Fi), 1)[0, 1].item()
            ps = []
            for c in range(Fv.shape[1]):
                Fr = Fi.clone(); Fr[:, c] = Fv[:, c]
                ps.append(torch.softmax(cls(Fr), 1)[0, 1].item())
            ps = np.asarray(ps)
            rows.append((name, p0, float(ps.min()), float(ps.max() - ps.min())))

    rows.sort(key=lambda r: -r[3])
    print("top 20 by range (name, p0, min, range):")
    for r in rows[:20]:
        print(f"  {r[0]}  p0={r[1]:.3f} min={r[2]:.3f} range={r[3]:.3f}")
    rng = np.array([r[3] for r in rows])
    print(f"scenes={len(rows)} range median={np.median(rng):.3f} "
          f"p90={np.percentile(rng,90):.3f} frac(range>0.1)={np.mean(rng>0.1):.3f}")


if __name__ == "__main__":
    main()
