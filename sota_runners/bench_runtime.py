"""Inference latency + parameter count for our rules and the classical baselines."""
import time

import numpy as np
import torch

from ssgf.config import get_default_config
from ssgf.fusion import MeanRule, SemanticPriorRule, SemanticSaliencyRule
from ssgf.models import Decoder, Encoder, build_prior
from ssgf.models.baselines import DenseFuse, IFCNN, fuse_l1
from ssgf.utils.checkpoint import load_ckpt


def count(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    dev = 'cuda'
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_ae.pt', encoder=enc, decoder=dec)
    prior = build_prior('tophat')
    mean_rule = MeanRule(enc, dec, cfg, dev)
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt('/root/autodl-tmp/ssgf/data/msrs_probe.pt', probe=probe_rule.probe)

    import numpy as np
    rng = np.random.RandomState(0)
    v = torch.from_numpy(rng.rand(1, 1, 384, 512).astype(np.float32)).to(dev)
    i = torch.from_numpy(rng.rand(1, 1, 384, 512).astype(np.float32)).to(dev)

    ours = count(dec) + count(enc)
    probe = count(probe_rule.probe)
    print('params: AE(enc+dec)=%.2fM probe=%.2fM total=%.2fM' % (ours, probe, ours + probe))
    for name, fn in [
        ('mean', lambda: mean_rule.fuse_pair(v.clone(), i.clone())),
        ('tophat', lambda: top_rule.fuse_pair(v.clone(), i.clone())),
        ('probe', lambda: probe_rule.fuse_pair(v.clone(), i.clone())),
        ('ensemble', lambda: (top_rule.fuse_pair(v.clone(), i.clone()),
                              probe_rule.fuse_pair(v.clone(), i.clone()))),
    ]:
        t = timeit(fn) * 1000
        print(f'{name:10s} {t:.1f} ms')

    for name, cls, ckpt in [('IFCNN', IFCNN, '/root/autodl-tmp/ssgf/data/btl/ifcnn.pt'),
                            ('DenseFuse', DenseFuse, '/root/autodl-tmp/ssgf/data/btl/densefuse.pt')]:
        m = cls().to(dev)
        m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False))
        m.eval()
        n = count(m)
        def fuse():
            with torch.no_grad():
                return m.decode(fuse_l1(m.encode(v * 2 - 1), m.encode(i * 2 - 1)))
        t = timeit(fuse)
        print(f'{name:10s} {t*1000:.1f} ms  params={n:.2f}M')


if __name__ == '__main__':
    main()
