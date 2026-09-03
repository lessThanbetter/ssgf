"""End-to-end fusion-aware training of the encoder + targetness rule + decoder + probe.

Keeps every module trainable (unlike BaseRule, which freezes them) and optimises the
fused image against BOTH sources (SSIM + MSE) plus probe distillation to the top-hat
teacher. This is the main lever on absolute VIF.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from ssgf.config import get_default_config
from ssgf.models import Decoder, Encoder, build_probe, build_prior
from ssgf.utils.checkpoint import save_ckpt
from ssgf.utils.io import find_pairs, imread


class PairDS(Dataset):
    def __init__(self, root, vis='vi', ir='ir', patch=192):
        self.pairs = find_pairs(root, vis, ir)
        self.patch = patch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        vp, ip = self.pairs[i]
        vs, ir = imread(vp), imread(ip)
        h, w = vs.shape
        y = np.random.randint(0, max(1, h - self.patch + 1))
        x = np.random.randint(0, max(1, w - self.patch + 1))
        vs = vs[y:y + self.patch, x:x + self.patch]
        ir = ir[y:y + self.patch, x:x + self.patch]
        return torch.from_numpy(vs).float().unsqueeze(0), torch.from_numpy(ir).float().unsqueeze(0)


def ssim_map(a, b, size=11, sigma=1.5):
    x = torch.arange(size, device=a.device).float() - (size // 2)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2)); k = torch.outer(g, g); k = k / k.sum()
    w = k.view(1, 1, size, size)
    c1, c2 = 0.01 ** 2, 0.03 ** 2; pad = size // 2
    mu1 = F.conv2d(a, w, padding=pad); mu2 = F.conv2d(b, w, padding=pad)
    s1 = F.conv2d(a * a, w, padding=pad) - mu1 * mu1
    s2 = F.conv2d(b * b, w, padding=pad) - mu2 * mu2
    s12 = F.conv2d(a * b, w, padding=pad) - mu1 * mu2
    return ((2 * s12 + c2) * (2 * mu1 * mu2 + c1)) / ((s1 + s2 + c2) * (mu1 * mu1 + mu2 * mu2 + c1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msrs', required=True)
    ap.add_argument('--out', default='data/msrs_e2e.pt')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--cfg', nargs='*', default=['backbone=wide'])
    args = ap.parse_args()
    over = {}
    for kv in args.cfg:
        k, _, v = kv.partition('='); over[k.strip()] = int(v) if v.isdigit() else v.strip()
    cfg = get_default_config(**over)
    cfg.decoder_tanh = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    enc, dec = Encoder(cfg).to(dev), Decoder(cfg).to(dev)
    probe = build_probe(cfg).to(dev)
    prior = build_prior('tophat')
    enc.train(); dec.train(); probe.train()
    params = list(enc.parameters()) + list(dec.parameters()) + list(probe.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    ds = PairDS(f'{args.msrs}/train')
    loader = DataLoader(ds, batch_size=12, shuffle=True, num_workers=0, drop_last=True)
    print(f'[e2e] {len(ds)} pairs, device={dev}')

    def forward(vis, ir):
        vn = vis * 2 - 1; inn = ir * 2 - 1
        Fv = enc(vn); Fi = enc(inn)
        act_v = Fv.abs().sum(1, keepdim=True); act_i = Fi.abs().sum(1, keepdim=True)
        s_v = torch.sigmoid(probe.map_logits(Fv)); s_i = torch.sigmoid(probe.map_logits(Fi))
        dv = s_v * act_v; di = s_i * act_i
        mx = torch.max(dv.max(), di.max()) + 1e-8
        tau = 0.3
        ev = torch.exp(dv / mx / tau); ei = torch.exp(di / mx / tau)
        wv = ev / (ev + ei); wi = ei / (ev + ei)
        F = wv * Fv + wi * Fi
        return (dec(F) + 1) / 2, wv, wi, Fv, Fi

    for ep in range(args.epochs):
        tot = n = 0
        for vis, ir in loader:
            vis, ir = vis.to(dev), ir.to(dev)
            f, wv, wi, Fv, Fi = forward(vis, ir)
            loss = (1 - ssim_map(f, vis).mean()) + (1 - ssim_map(f, ir).mean()) + \
                   5.0 * (F.mse_loss(f, vis) + F.mse_loss(f, ir))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f'epoch {ep+1}/{args.epochs} loss={tot/max(n,1):.4f}')
        save_ckpt(args.out, enc, dec, probe=probe, cfg=cfg, meta={'phase': 'e2e-fusion'})
    print(f'saved -> {args.out}')


def _01(t):
    return t.detach().cpu().numpy()[0]


if __name__ == '__main__':
    main()
