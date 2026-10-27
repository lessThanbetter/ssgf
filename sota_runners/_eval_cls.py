"""Sanity-check a modality classifier on MSRS train patches (server-side)."""

import glob
import random
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/ssgf")
from ssgf.config import get_default_config
from ssgf.models import Decoder, Encoder, ModalityClassifier
from ssgf.utils.checkpoint import load_ckpt


def main():
    ae = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/ssgf/data/msrs_ae.pt"
    ck = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/ssgf/data/msrs_csf.pt"
    root = "/root/autodl-tmp/datasets/MSRS-main/train"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()

    # compare the encoder bundled in the classifier ckpt vs the standalone ae ckpt
    ae_pay = torch.load(ae, map_location="cpu")
    ck_pay = torch.load(ck, map_location="cpu")
    if "encoder" in ck_pay and ck_pay["encoder"] is not None:
        diffs = [(k, float((ae_pay["encoder"][k].float()
                            - ck_pay["encoder"][k].float()).abs().max()))
                 for k in ae_pay["encoder"]]
        print("encoder max|ae - csf| per-param (first 5):", diffs[:5])
        print("encoder global max diff:",
              max(d for _, d in diffs))

    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(ae, encoder=enc, decoder=dec)
    enc.to(dev).eval()
    enc2 = Encoder(cfg)
    load_ckpt(ck, encoder=enc2)
    enc2.to(dev).eval()
    cls = ModalityClassifier(cfg).to(dev).eval()
    load_ckpt(ck, classifier=cls)

    vis = sorted(glob.glob(f"{root}/vi/*"))[:300]
    ir = sorted(glob.glob(f"{root}/ir/*"))[:300]
    rng = random.Random(0)

    def patch(p):
        im = cv2.imread(p, 0)
        if im is None:
            return None
        h, w = im.shape
        if h < 128 or w < 128:
            return None
        y, x = rng.randint(0, h - 128), rng.randint(0, w - 128)
        return im[y:y + 128, x:x + 128].astype(np.float32) / 255.0

    def report(paths, label, name, encoder):
        n = correct = 0
        pir = []
        for p in paths:
            im = patch(p)
            if im is None:
                continue
            t = torch.from_numpy(im * 2 - 1)[None, None].to(dev)
            with torch.no_grad():
                pr = torch.softmax(cls(encoder(t)), 1)[0]
            pir.append(pr[1].item())
            correct += int(int(pr.argmax().item()) == label)
            n += 1
        print(f"{name}: n={n} acc={correct / max(n, 1):.3f} mean_P(IR)={np.mean(pir):.3f}")

    print("-- encoder from msrs_ae.pt --")
    report(vis, 0, "MSRS visible", enc)
    report(ir, 1, "MSRS infrared", enc)
    print("-- encoder bundled in msrs_csf.pt --")
    report(vis, 0, "MSRS visible", enc2)
    report(ir, 1, "MSRS infrared", enc2)


if __name__ == "__main__":
    main()
