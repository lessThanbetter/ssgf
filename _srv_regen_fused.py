"""Regenerate the fused inputs that are no longer on disk.

Two groups are rebuilt at native resolution for the 300 LLVIP test pairs, in the
locations the released analysis scripts expect:

  data/k1_native/imgs/<control>/   the eight controls of k1_controls.py
  sota_outs/csf_llvip_native/      the classification-saliency rule of j4_csf.py

Nothing is overwritten: an existing file is skipped. The regenerated mean, top-hat
and probe images are compared against the copies kept under datasets/det/eval_fused
so that the reconstruction can be checked rather than assumed.
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/ssgf")

from ssgf.config import get_default_config  # noqa: E402
from ssgf.fusion import ModalityCSRule  # noqa: E402
from ssgf.models import Decoder, Encoder  # noqa: E402
from ssgf.utils.checkpoint import load_ckpt  # noqa: E402
from ssgf.utils.io import imread  # noqa: E402
from sota_runners.k1_controls import CONTROLS, build_rules  # noqa: E402

BASE = "/root/autodl-tmp"
TEST = f"{BASE}/datasets/det/test"
EF = f"{BASE}/datasets/det/eval_fused"
AE = f"{BASE}/ssgf/data/msrs_ae.pt"
CSF = f"{BASE}/ssgf/data/msrs_csf.pt"
K1 = f"{BASE}/ssgf/data/k1_native/imgs"
CSF_OUT = f"{BASE}/ssgf/sota_outs/csf_llvip_native"
CHECK = ["mean", "tophat", "probe"]


def main():
    dev = "cuda"
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(AE, encoder=enc, decoder=dec)
    rules = build_rules(cfg, enc, dec, dev)
    csf = ModalityCSRule(enc, dec, cfg, dev)
    payload = load_ckpt(CSF, classifier=csf.scorer)
    if payload.get("classifier") is None:
        raise RuntimeError("msrs_csf.pt has no classifier weights")

    for c in CONTROLS:
        os.makedirs(os.path.join(K1, c), exist_ok=True)
    os.makedirs(CSF_OUT, exist_ok=True)

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{TEST}/labels"))
    print(f"[regen] {len(stems)} pairs", flush=True)
    t0 = time.time()
    for i, stem in enumerate(stems):
        v = imread(f"{TEST}/visible/{stem}.jpg")
        r = imread(f"{TEST}/infrared/{stem}.jpg")
        vt = torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(dev)
        rt = torch.from_numpy(r).unsqueeze(0).unsqueeze(0).to(dev)
        for c in CONTROLS:
            dst = os.path.join(K1, c, stem + ".png")
            if os.path.exists(dst):
                continue
            with torch.no_grad():
                f = rules[c].fuse_pair(vt, rt)[0]
            fu = np.clip(f.squeeze().cpu().numpy(), 0, 1)
            Image.fromarray((fu * 255).astype("uint8")).save(dst)
        dst = os.path.join(CSF_OUT, stem + ".png")
        if not os.path.exists(dst):
            f = csf.fuse_pair(vt, rt)[0]
            fu = np.clip(f[0, 0].detach().cpu().numpy(), 0, 1)
            Image.fromarray((fu * 255).astype("uint8")).save(dst)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(stems)}  {el:.0f}s ({el/(i+1):.2f}s/img)", flush=True)

    print("== consistency check against the kept copies ==", flush=True)
    for c in CHECK:
        diffs = []
        for stem in stems[:10]:
            new = os.path.join(K1, c, stem + ".png")
            old = os.path.join(EF, c, "images", stem + ".png")
            if not (os.path.exists(new) and os.path.exists(old)):
                old = os.path.join(EF, c, "images", stem + ".jpg")
                if not os.path.exists(old):
                    continue
            a = np.asarray(Image.open(new).convert("L"), np.float32)
            b = np.asarray(Image.open(old).convert("L"), np.float32)
            if a.shape != b.shape:
                diffs.append(float("nan"))
                continue
            diffs.append(float(np.abs(a - b).mean()))
        print("  %-10s mean |difference| over 10 stems: %s" % (c, np.round(diffs, 2)), flush=True)
    for c in CONTROLS + ["csf"]:
        d = os.path.join(K1, c) if c != "csf" else CSF_OUT
        print("  %-16s %d files" % (c, len(os.listdir(d))), flush=True)
    print("REGEN_DONE", flush=True)


if __name__ == "__main__":
    main()
