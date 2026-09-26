"""Standard image-fusion metrics.

Convention: arrays are ``np.float`` grayscale in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity as _ssim


def _to_gray(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        x = x[..., 0] if x.shape[-1] == 1 else x.mean(axis=-1)
    return x.astype(np.float32)


def entropy(x: np.ndarray) -> float:
    x = _to_gray(x)
    hist, _ = np.histogram(x, bins=256, range=(0, 1))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def std(x: np.ndarray) -> float:
    return float(_to_gray(x).std())


def spatial_frequency(x: np.ndarray) -> float:
    x = _to_gray(x)
    df = np.diff(x, axis=0)
    dc = np.diff(x, axis=1)
    rf = np.sqrt(np.mean(df ** 2))
    cf = np.sqrt(np.mean(dc ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def average_gradient(x: np.ndarray) -> float:
    x = _to_gray(x)
    gy, gx = np.gradient(x)
    return float(np.sqrt((gx ** 2 + gy ** 2).mean()))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((_to_gray(a) - _to_gray(b)) ** 2)))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((_to_gray(a) - _to_gray(b)) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(_ssim(_to_gray(a), _to_gray(b), data_range=1.0))


def ms_ssim(a: np.ndarray, b: np.ndarray) -> float:
    # skimage's SSIM is a single-scale version here; use multi-scale by down-sampling.
    a, b = _to_gray(a), _to_gray(b)
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = []
    for _ in range(len(weights)):
        levels.append(_ssim(a, b, data_range=1.0))
        a = ndi.zoom(a, 0.5, order=1)
        b = ndi.zoom(b, 0.5, order=1)
    return float(np.prod(np.array(levels) ** weights))


def _gauss_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size) - (size // 2)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def qabf(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    """Edge-based fusion quality (Xydeas & Petrovic)."""
    a, b, f = _to_gray(a), _to_gray(b), _to_gray(f)
    K = np.array([1 / 8, 1 / 8, 1 / 8, 1 / 8, -1, 1 / 8, 1 / 8, 1 / 8, 1 / 8]).reshape(3, 3)

    def _grad(x):
        gx = ndi.convolve(x, K, mode="nearest")
        gy = ndi.convolve(x, K.T, mode="nearest")
        return np.sqrt(gx ** 2 + gy ** 2), np.arctan2(gy, gx)

    ga, aa = _grad(a)
    gb, ab = _grad(b)
    gf, af = _grad(f)
    Tg, Ag, kg, ka = 0.9994, 0.9879, -5, -10
    Ga = Tg / (1 + np.exp(kg * (ga - Ag)))
    Gb = Tg / (1 + np.exp(kg * (gb - Ag)))
    Aa = 2 / (1 + np.exp(ka * (np.abs(aa - af) - Ag)))
    Ab = 2 / (1 + np.exp(ka * (np.abs(ab - af) - Ag)))
    Qaf = Ga * Aa
    Qbf = Gb * Ab
    w = ga + gb
    total = np.sum(w)
    if total < 1e-12:
        return 0.0
    return float(np.sum(w * (Qaf + Qbf)) / total)


def scd(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> float:
    """Structural correlation difference (SCD)."""
    a, b, f = _to_gray(a), _to_gray(b), _to_gray(f)
    fa = np.sum((f - f.mean()) * (a - a.mean())) / (np.sqrt(np.sum((f - f.mean()) ** 2)) * np.sqrt(np.sum((a - a.mean()) ** 2)) + 1e-9)
    fb = np.sum((f - f.mean()) * (b - b.mean())) / (np.sqrt(np.sum((f - f.mean()) ** 2)) * np.sqrt(np.sum((b - b.mean()) ** 2)) + 1e-9)
    return float(fa + fb)


def vifp(ref: np.ndarray, dist: np.ndarray, sigma_nsq: float = 2.0) -> float:
    """Single-image pixel VIF between a reference and a distorted image.

    This is the variant used for the reported tables. It accumulates all four
    Gaussian widths on the full-resolution grid and keeps ``sigma_nsq`` at 2.0,
    which is the reference value for 8-bit input, while the arrays here are in
    [0, 1]. ``vifp_reference`` below implements the reference construction with
    decimation between scales and a noise variance expressed in the units of the
    input; the two differ, and Sec. 3 of the paper and Supplementary Sec. 8 report
    the consequence.
    """
    ref, dist = _to_gray(ref), _to_gray(dist)
    num, den = 0.0, 0.0
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        sd = 2.0 ** (scale - 1)
        g = _gauss_kernel(n, sd / 5.0 * 5)
        if scale > 1:
            ref = ndi.convolve(ref, g, mode="nearest")
            dist = ndi.convolve(dist, g, mode="nearest")
        mu1 = ndi.convolve(ref, g, mode="nearest")
        mu2 = ndi.convolve(dist, g, mode="nearest")
        s11 = ndi.convolve(ref * ref, g, mode="nearest") - mu1 * mu1
        s22 = ndi.convolve(dist * dist, g, mode="nearest") - mu2 * mu2
        s12 = ndi.convolve(ref * dist, g, mode="nearest") - mu1 * mu2
        g_map = s12 / (s11 + 1e-9)
        sv_sq = s22 - g_map * s12
        g_map = np.clip(g_map, 0, 1)
        sv_sq[sv_sq < 0] = 0
        num += np.sum(np.log10(1 + g_map ** 2 * s11 / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + s11 / sigma_nsq))
    return float(num / (den + 1e-9))


def vifp_reference(ref: np.ndarray, dist: np.ndarray) -> float:
    """Single-image pixel VIF in the reference construction.

    Differences from :func:`vifp`: the Gaussian width at scale ``s`` is
    ``window / 5`` rather than ``2**(s-1)``, both images are decimated by two
    between scales so that each scale is accumulated at its own resolution, and the
    noise variance is 2.0 expressed in the units of the input, that is 2 / 255**2
    for arrays in [0, 1].
    """
    ref, dist = _to_gray(ref).copy(), _to_gray(dist).copy()
    sigma_nsq = 2.0 / (255.0 ** 2)
    num, den = 0.0, 0.0
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        g = _gauss_kernel(n, n / 5.0)
        if scale > 1:
            ref = ndi.convolve(ref, g, mode="nearest")[::2, ::2]
            dist = ndi.convolve(dist, g, mode="nearest")[::2, ::2]
        mu1 = ndi.convolve(ref, g, mode="nearest")
        mu2 = ndi.convolve(dist, g, mode="nearest")
        s11 = ndi.convolve(ref * ref, g, mode="nearest") - mu1 * mu1
        s22 = ndi.convolve(dist * dist, g, mode="nearest") - mu2 * mu2
        s12 = ndi.convolve(ref * dist, g, mode="nearest") - mu1 * mu2
        g_map = np.clip(s12 / (s11 + 1e-9), 0, 1)
        sv_sq = s22 - g_map * s12
        sv_sq[sv_sq < 0] = 0
        num += np.sum(np.log10(1 + g_map ** 2 * s11 / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + s11 / sigma_nsq))
    return float(num / (den + 1e-9))


def report(fused: np.ndarray, vis: np.ndarray, ir: np.ndarray) -> dict:
    """Compute the metric suite for one fused output."""
    f, a, b = _to_gray(fused), _to_gray(vis), _to_gray(ir)
    d = {
        "EN": entropy(f),
        "VIF_ref": (vifp_reference(a, f) + vifp_reference(b, f)) / 2,
        "SD": std(f),
        "SF": spatial_frequency(f),
        "AG": average_gradient(f),
        "PSNR_vis": psnr(f, a),
        "PSNR_ir": psnr(f, b),
        "SSIM_vis": ssim(f, a),
        "SSIM_ir": ssim(f, b),
        "MS_SSIM": ms_ssim(f, a),
        "Qabf": qabf(a, b, f),
        "SCD": scd(a, b, f),
        "VIF_vis": vifp(a, f),
        "VIF_ir": vifp(b, f),
    }
    d["VIF"] = (d["VIF_vis"] + d["VIF_ir"]) / 2
    return d
