"""Shared CSF (Xu et al., IEEE TCI 2021) figure style.

The reference paper draws block diagrams as flat, lightly-rounded boxes,
feature maps as tilted "card fan" stacks colored by modality, a pale purple
container for the saliency/weighting rule, and Times-like math labels.  This
module encodes that palette + a few patches so the user's figures can be
recreated in the same visual language.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 12,
        "axes.unicode_minus": False,
    }
)


# ---- palette sampled from the CSF paper figures ----
ORANGE = "#FDB786"      # visible feature maps
ORANGE_EDGE = "#E8935A"
BLUE = "#BACEEF"        # infrared feature maps
BLUE_EDGE = "#6C93C4"
GREEN = "#CCE7BA"       # fused feature maps
GREEN_EDGE = "#9FC77F"
PURPLE = "#E9DFF3"      # saliency/weighting rule container
PURPLE_EDGE = "#8B5FA8"
WHITE = "#FFFFFF"
INK = "#1a1a1a"
GRAY = "#4a4a4a"


def box(ax, x, y, w, h, text="", fc=WHITE, ec=INK, fs=12, tc=INK, lw=1.2,
        rounding=0.10, italic=False):
    """Flat, lightly-rounded box with centered text (CSF style)."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=4,
        )
    )
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=5, linespacing=1.25,
                fontstyle="italic" if italic else "normal")
    return (x, y, w, h)


def arrow(ax, p0, p1, color=INK, lw=1.3, rad=0.0, ms=13):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, color=color, zorder=3,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=1, shrinkB=1)
    )


def feature_fan(ax, x, y, w, h, n=5, fc=ORANGE, ec=ORANGE_EDGE, dx=0.10, dy=0.06,
                shear=0.22, lw=0.8, label=None, fs=11, tc=INK):
    """Draw n tilted card fans (stacked parallelograms), front card last."""
    pts = []
    for k in range(n - 1, -1, -1):
        ox, oy = k * dx, k * dy
        p = [
            (x + ox, y + oy),
            (x + w + ox, y + oy),
            (x + w + ox + shear, y + h + oy),
            (x + ox + shear, y + h + oy),
        ]
        ax.add_patch(Polygon(p, closed=True, facecolor=fc, edgecolor=ec,
                             linewidth=lw, zorder=2, joinstyle="miter"))
        pts.append(p)
    if label:
        front = pts[-1]
        cx = (front[0][0] + front[2][0]) / 2
        ax.text(cx, y + h / 2, label, ha="center", va="center", fontsize=fs,
                color=tc, zorder=6, fontweight="bold")
    return pts


def thumb(ax, img, x, y, w, h, edge=INK, lw=1.2):
    """Show a grayscale thumbnail inset, tightly cropped, CSF style."""
    from matplotlib import colors

    ax.imshow(img, extent=(x, x + w, y, y + h), aspect="auto",
              cmap="gray", zorder=3, interpolation="bilinear")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=edge,
                           linewidth=lw, zorder=4))


def gradient_thumb(hw=160, hh=120, phase=0.0, seed=0):
    """Synthetic grayscale scene used when no real image is available."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:hh, 0:hw]
    base = 0.35 + 0.25 * np.sin(xx / hw * np.pi + phase) \
        + 0.20 * np.cos(yy / hh * 2.2 * np.pi)
    blob = np.exp(-(((xx - hw * (0.30 + 0.4 * phase)) ** 2
                     + (yy - hh * 0.55) ** 2) / (2 * (0.09 * hw) ** 2)))
    noise = rng.normal(0, 0.03, (hh, hw))
    img = np.clip(base + 0.5 * blob + noise, 0, 1)
    return img


def read_gray(path, hw=160, hh=120):
    """Read an image and return a small grayscale array (or None)."""
    from PIL import Image

    try:
        im = Image.open(path).convert("L")
    except Exception:
        return None
    im = im.resize((hw, hh), Image.LANCZOS)
    return np.asarray(im, dtype=float) / 255.0


def new_fig(w, h):
    fig, ax = matplotlib.pyplot.subplots(figsize=(w, h), dpi=300)
    ax.set_axis_off()
    return fig, ax


def save(fig, path_stem):
    for ext, kw in {
        "png": dict(dpi=300),
        "svg": dict(),
        "pdf": dict(),
    }.items():
        fig.savefig(f"{path_stem}.{ext}", bbox_inches="tight", facecolor="white", **kw)
