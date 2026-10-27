"""Which input normalization does the encoder expect: [0,1] or [-1,1]?"""

import glob

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, "/root/autodl-tmp/ssgf")
from ssgf.config import get_default_config
from ssgf.models import Decoder, Encoder, ModalityClassifier
from ssgf.utils.checkpoint import load_ckpt


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt("/root/autodl-tmp/ssgf/data/msrs_ae.pt", encoder=enc, decoder=dec)
    enc, dec = enc.to(dev).eval(), dec.to(dev).eval()
    cls = ModalityClassifier(cfg).to(dev).eval()
    load_ckpt("/root/autodl-tmp/ssgf/data/msrs_csf.pt", encoder=enc, classifier=cls)

    root = "/root/autodl-tmp/datasets/MSRS-main/train"
    vis = sorted(glob.glob(f"{root}/vi/*"))[:120]
    ir = sorted(glob.glob(f"{root}/ir/*"))[:120]
    rng = np.random.default_rng(0)

    def patches(paths):
        out = []
        for p in paths:
            im = cv2.imread(p, 0)
            if im is None or im.shape[0] < 128 or im.shape[1] < 128:
                continue
            y, x = rng.integers(0, im.shape[0] - 128), rng.integers(0, im.shape[1] - 128)
            out.append(im[y:y + 128, x:x + 128].astype(np.float32) / 255.0)
        return out

    for tag, f in (("msrs_ae.pt", "/root/autodl-tmp/ssgf/data/msrs_ae.pt"),):
        e, d = Encoder(cfg), Decoder(cfg)
        load_ckpt(f, encoder=e, decoder=d)
        e, d = e.to(dev).eval(), d.to(dev).eval()
        ranges = []
        for norm in ("01", "m11"):
            errs = []
            for im in patches(ir):
                t = im * (2 if norm == "m11" else 1) - (1 if norm == "m11" else 0)
                tt = torch.from_numpy(t)[None, None].to(dev)
                with torch.no_grad():
                    rec = d(e(tt))[0, 0].cpu().numpy()
                tgt = t if norm == "m11" else im
                errs.append(float(np.mean((rec - tgt) ** 2)))
                ranges.append((float(rec.min()), float(rec.max())))
            print(f"recon MSE  norm={norm:>3}  ({tag})  = {np.mean(errs):.5f}")
        print("decoder output range (min,max) samples:", ranges[:3])

    print("-- classifier (encoder+head from msrs_csf.pt) --")
    for norm in ("01", "m11"):
        accs = {}
        for paths, lab in ((vis, 0), (ir, 1)):
            c = n = 0
            for im in patches(paths):
                t = im * (2 if norm == "m11" else 1) - (1 if norm == "m11" else 0)
                with torch.no_grad():
                    pr = torch.softmax(cls(enc(torch.from_numpy(t)[None, None].to(dev))), 1)[0]
                c += int(int(pr.argmax()) == lab)
                n += 1
            accs[lab] = c / max(n, 1)
        print(f"classifier acc  norm={norm:>3}  vis={accs[0]:.3f}  ir={accs[1]:.3f}")


if __name__ == "__main__":
    main()
