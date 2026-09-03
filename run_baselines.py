"""Train and evaluate IFCNN / DenseFuse baselines on the same data.

Training is self-supervised reconstruction on MSRS (identical loss to our AE) so the
comparison is fair. Evaluation fuses test pairs with l1-norm fusion and reports the
same metrics.

Usage (server):
    python run_baselines.py --mode train --msrs /root/autodl-tmp/datasets/MSRS-main --out data/btl
    python run_baselines.py --mode eval  --msrs /root/autodl-tmp/datasets/MSRS-main --ckpt data/btl
"""

from __future__ import annotations

import argparse, os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ssgf.config import get_default_config
from ssgf.data import FusionDataset
from ssgf.models.baselines import IFCNN, DenseFuse, fuse_l1
from ssgf.train import ssim_map
from ssgf.utils.metrics import report
from ssgf.utils.io import imread, find_pairs


def _to01(x):
    return (x + 1) / 2.0


def _rz(vis, ir, max_side=512):
    h, w = vis.shape
    sc = max_side / max(h, w)
    if sc >= 1:
        return vis, ir
    nw, nh = int(w * sc), int(h * sc)
    rz = lambda a: np.asarray(Image.fromarray((a * 255).astype("uint8")).resize((nw, nh)), np.float32) / 255.0
    return rz(vis), rz(ir)


def train_model(model, msrs, out_path, epochs):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).train()
    ds = FusionDataset(f"{msrs}/train", "vi", "ir", patch_size=128, train=True)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        tot = n = 0
        for batch in loader:
            x = batch.to(dev)
            xn = x * 2 - 1
            recon = model.decode(model.encode(xn))
            loss = 25 * torch.mean((xn - recon) ** 2) + (1 - ssim_map(xn, recon))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep+1}/{epochs} loss={tot/max(n,1):.4f}")
    torch.save(model.state_dict(), out_path)
    print(f"saved -> {out_path}")


def fuse_pair(model, vis, ir, dev):
    v = (torch.from_numpy(vis) * 2 - 1).unsqueeze(0).unsqueeze(0).to(dev)
    i = (torch.from_numpy(ir) * 2 - 1).unsqueeze(0).unsqueeze(0).to(dev)
    fv, fi = model.encode(v), model.encode(i)
    fused = model.decode(fuse_l1(fv, fi))
    return _to01(fused.squeeze()).detach().cpu().numpy()


def eval_models(ckpt_dir, msrs, dev):
    models = {"ifcnn": IFCNN(), "densefuse": DenseFuse()}
    for k, m in models.items():
        m.to(dev).eval()
        m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{k}.pt"), map_location="cpu"))
    ds_spec = [("TNO", "data/tno_trial", "vis", "ir", 0),
               ("MSRS", f"{msrs}/test", "vi", "ir", 0),
               ("RoadScene", "/root/autodl-tmp/datasets/RoadScene-master", "crop_HR_visible", "infrared", 0)]
    for dname, root, vd, idd, lim in ds_spec:
        pairs = find_pairs(root, vd, idd)
        for k, m in models.items():
            vif = []
            for vp, ip in pairs:
                vis, ir = imread(vp), imread(ip)
                h, w = min(vis.shape[0], ir.shape[0]), min(vis.shape[1], ir.shape[1])
                vis, ir = vis[:h, :w], ir[:h, :w]
                vis, ir = _rz(vis, ir)
                f = fuse_pair(m, vis, ir, dev)
                vif.append(report(f, vis, ir)["VIF"])
            print(f"{dname:10s} {k:10s} VIF={np.mean(vif):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--msrs", required=True)
    ap.add_argument("--out", default="data/btl")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.mode == "train":
        train_model(IFCNN(), args.msrs, os.path.join(args.out, "ifcnn.pt"), args.epochs)
        train_model(DenseFuse(), args.msrs, os.path.join(args.out, "densefuse.pt"), args.epochs)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        eval_models(args.ckpt or args.out, args.msrs, dev)


if __name__ == "__main__":
    main()
