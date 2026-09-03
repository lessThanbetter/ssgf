"""Report probe-rule VIF using the end-to-end fusion-trained (wide) model."""
import numpy as np
import torch
from PIL import Image

from ssgf.config import get_default_config
from ssgf.fusion import SemanticSaliencyRule
from ssgf.models import Decoder, Encoder
from ssgf.utils.checkpoint import load_ckpt
from ssgf.utils.io import find_pairs, imread
from ssgf.utils.metrics import vifp


DATASETS = [
    ('TNO', '/root/autodl-tmp/ssgf/data/tno_trial', 'vis', 'ir'),
    ('MSRS', '/root/autodl-tmp/datasets/MSRS-main/test', 'vi', 'ir'),
    ('RoadScene', '/root/autodl-tmp/datasets/RoadScene-master', 'crop_HR_visible', 'infrared'),
]


def rz(vis, ir, max_side=512):
    h, w = vis.shape
    sc = max_side / max(h, w)
    if sc >= 1:
        return vis, ir
    nw, nh = int(w * sc), int(h * sc)
    z = lambda a: np.asarray(Image.fromarray((a * 255).astype('uint8')).resize((nw, nh)), np.float32) / 255.0
    return z(vis), z(ir)


def main():
    cfg = get_default_config(backbone='wide'); cfg.decoder_tanh = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc, dec = Encoder(cfg), Decoder(cfg)
    rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_e2e.pt', encoder=enc, decoder=dec, probe=rule.probe)
    for name, root, vd, ird in DATASETS:
        pairs = find_pairs(root, vd, ird)
        vals = []
        for vp, ip in pairs:
            vis, ir = rz(imread(vp), imread(ip))
            v = torch.from_numpy(vis).to(dev).unsqueeze(0).unsqueeze(0)
            i = torch.from_numpy(ir).to(dev).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                f = rule.fuse_pair(v, i)[0].squeeze().cpu().numpy()
            f = np.asarray(Image.fromarray((np.clip(f, 0, 1) * 255).astype('uint8')).resize((vis.shape[1], vis.shape[0])), np.float32) / 255.0
            vals.append((vifp(vis, f) + vifp(ir, f)) / 2)
        print(f'{name}: VIF={np.mean(vals):.4f} (n={len(vals)})')


if __name__ == '__main__':
    main()
