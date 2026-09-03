"""Downstream semantic-segmentation (mIoU) evaluation on fused images.

Protocol: train a fixed U-Net on MSRS-train source images, then apply it to each
fusion method's outputs on MSRS-test and report mIoU against the ground-truth
segmentation labels. Higher mIoU on fused images => better fusion for downstream.

Usage (server):
    python seg_downstream.py --mode train --msrs /root/autodl-tmp/datasets/MSRS-main \
        --out data/segnet.pt --epochs 30
    python seg_downstream.py --mode eval --msrs /root/autodl-tmp/datasets/MSRS-main \
        --model data/segnet.pt --ae data/msrs_ae.pt --probe data/msrs_probe.pt
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, SemanticProbe, build_prior
from ssgf.models.segnet import UNet
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import find_pairs, imread


N_CLASSES = 9
SIZE = 256


def _gray(path):
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0


class SegTrainDS(Dataset):
    def __init__(self, msrs, split="train", patch=SIZE):
        self.labels = sorted(os.listdir(f"{msrs}/{split}/Segmentation_labels"))
        self.root = f"{msrs}/{split}"
        self.patch = patch

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        name = self.labels[i]
        # sample either modality (registered pair -> same label) so the task net is
        # modality-agnostic; fused images should then segment at least as well as either source.
        mod = random.choice(["vi", "ir"])
        img = _gray(f"{self.root}/{mod}/{name}")
        lab = np.asarray(Image.open(f"{self.root}/Segmentation_labels/{name}"), dtype=np.int64)
        h, w = img.shape
        p = self.patch
        if h >= p and w >= p:
            y, x = random.randint(0, h - p), random.randint(0, w - p)
            img, lab = img[y : y + p, x : x + p], lab[y : y + p, x : x + p]
        else:
            img = np.asarray(Image.fromarray((img * 255).astype("uint8")).resize((p, p)), np.float32) / 255.0
            lab = np.asarray(Image.fromarray(lab.astype("uint8")).resize((p, p), Image.NEAREST), np.int64)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(lab).long()


def train(args):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_ch=1, n_classes=N_CLASSES).to(dev)
    ds = SegTrainDS(args.msrs, "train")
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[seg] train {len(ds)} images, dev={dev}")
    for ep in range(args.epochs):
        tot = n = 0
        for img, lab in loader:
            img, lab = img.to(dev), lab.to(dev)
            logits = model(img)
            loss = F.cross_entropy(logits, lab)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep+1}/{args.epochs} ce={tot/max(n,1):.4f}")
    torch.save(model.state_dict(), args.out)
    print(f"[seg] saved -> {args.out}")


def _mIoU(pred, gt, n_classes):
    pred = np.clip(pred, 0, n_classes - 1)
    ious = []
    for c in range(n_classes):
        p = pred == c; g = gt == c
        inter = (p & g).sum().item(); union = (p | g).sum().item()
        ious.append(inter / (union + 1e-9))
    return np.mean(ious)


def eval_metric(model, img, gt, dev):
    with torch.no_grad():
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(dev)
        b = min(gt.shape[0], gt.shape[1])
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        logits = model(x)
        pred = logits.argmax(1)[0].cpu().numpy()
    # resize gt to SIZE
    gt_r = np.asarray(Image.fromarray(gt.astype("uint8")).resize((SIZE, SIZE), Image.NEAREST), np.int64)
    return _mIoU(pred, gt_r, N_CLASSES)


def fused_img(rule, vis, ir):
    v = torch.from_numpy(vis).unsqueeze(0).unsqueeze(0)
    i = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0)
    f, _ = rule.fuse_pair(v, i)
    return f.squeeze().cpu().numpy()


def evaluate(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model = UNet(in_ch=1, n_classes=N_CLASSES).to(dev)
    seg_model.load_state_dict(torch.load(args.model, map_location="cpu"))
    seg_model.eval()
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae, encoder=enc, decoder=dec)
    prior = build_prior("tophat")
    mean_rule = MeanRule(enc, dec, cfg, dev)
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt(args.probe, probe=probe_rule.probe)

    names = ["vis", "ir", "mean", "tophat", "probe", "ensemble"]
    acc = {k: [] for k in names}
    msrs = args.msrs
    pairs = find_pairs(f"{msrs}/test", "vi", "ir")
    labels = {os.path.splitext(os.path.basename(v))[0]: None for v, _ in pairs}
    print(f"[seg-eval] {len(pairs)} pairs")

    for vis_path, ir_path in pairs:
        vis = _gray(vis_path); ir = _gray(ir_path)
        name = [k for k in labels if k in vis_path][0]
        gt = np.asarray(Image.open(f"{msrs}/test/Segmentation_labels/{name}.png"), dtype=np.int64)
        acc["vis"].append(eval_metric(seg_model, vis, gt, dev))
        acc["ir"].append(eval_metric(seg_model, ir, gt, dev))
        f_mean = fused_img(mean_rule, vis, ir)
        f_top = fused_img(top_rule, vis, ir)
        f_probe = fused_img(probe_rule, vis, ir)
        f_ens = (f_top + f_probe) / 2.0
        acc["mean"].append(eval_metric(seg_model, f_mean, gt, dev))
        acc["tophat"].append(eval_metric(seg_model, f_top, gt, dev))
        acc["probe"].append(eval_metric(seg_model, f_probe, gt, dev))
        acc["ensemble"].append(eval_metric(seg_model, f_ens, gt, dev))

    print("\n=== downstream segmentation mIoU (MSRS-test, 8 classes) ===")
    for k in names:
        a = np.array(acc[k])
        print(f"{k.ljust(9)} mIoU={a.mean():.4f}  +/- {a.std():.4f}" if k != "ensemble" else f"{k.ljust(9)} mIoU={a.mean():.4f}  +/- {a.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--msrs", required=True)
    ap.add_argument("--out", default="data/segnet.pt")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ae", default="data/msrs_ae.pt")
    ap.add_argument("--probe", default="data/msrs_probe.pt")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
