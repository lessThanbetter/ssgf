"""Inference latency, parameter count and extra peak GPU memory."""
import argparse
import time
from pathlib import Path

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


def peak_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del result
    return peak / (1024 ** 2)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-checkpoint", default=str(root / "data/msrs_ae.pt"))
    parser.add_argument("--probe-checkpoint", default=str(root / "data/msrs_probe.pt"))
    parser.add_argument("--ifcnn-checkpoint", default=str(root / "data/btl/ifcnn.pt"))
    parser.add_argument("--densefuse-checkpoint", default=str(root / "data/btl/densefuse.pt"))
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("CUDA is required for the memory benchmark")
    cfg = get_default_config()
    enc, dec = Encoder(cfg), Decoder(cfg)
    load_ckpt(args.ae_checkpoint, encoder=enc, decoder=dec)
    prior = build_prior('tophat')
    mean_rule = MeanRule(enc, dec, cfg, dev)
    top_rule = SemanticPriorRule(enc, dec, cfg, prior, dev)
    probe_rule = SemanticSaliencyRule(enc, dec, cfg, dev)
    load_ckpt(args.probe_checkpoint, probe=probe_rule.probe)

    import numpy as np
    rng = np.random.RandomState(0)
    v = torch.from_numpy(rng.rand(1, 1, 384, 512).astype(np.float32)).to(dev)
    i = torch.from_numpy(rng.rand(1, 1, 384, 512).astype(np.float32)).to(dev)

    backbone = count(dec) + count(enc)
    probe_params = count(probe_rule.probe)
    print("device=%s input=1x1x384x512 FP32 batch=1" % torch.cuda.get_device_name(0))
    print("%-10s %10s %10s %16s" % ("method", "params(M)", "time(ms)", "extra_peak(MiB)"))
    for name, params, fn in [
        ('mean', backbone, lambda: mean_rule.fuse_pair(v.clone(), i.clone())),
        ('tophat', backbone, lambda: top_rule.fuse_pair(v.clone(), i.clone())),
        ('probe', backbone + probe_params,
         lambda: probe_rule.fuse_pair(v.clone(), i.clone())),
        ('ensemble', backbone + probe_params,
         lambda: (top_rule.fuse_pair(v.clone(), i.clone()),
                  probe_rule.fuse_pair(v.clone(), i.clone()))),
    ]:
        t = timeit(fn) * 1000
        memory = peak_mb(fn)
        print(f"{name:10s} {params:10.2f} {t:10.1f} {memory:16.1f}")

    for name, cls, ckpt in [
        ('IFCNN', IFCNN, args.ifcnn_checkpoint),
        ('DenseFuse', DenseFuse, args.densefuse_checkpoint),
    ]:
        m = cls().to(dev)
        m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False))
        m.eval()
        n = count(m)
        def fuse():
            with torch.no_grad():
                return m.decode(fuse_l1(m.encode(v * 2 - 1), m.encode(i * 2 - 1)))
        t = timeit(fuse)
        memory = peak_mb(fuse)
        print(f"{name:10s} {n:10.2f} {t * 1000:10.1f} {memory:16.1f}")


if __name__ == '__main__':
    main()
