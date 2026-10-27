"""Shared plotting style for the manuscript figures.

Modelled on the reference manuscript the authors supplied: Times-family serif text,
a full box frame with a light dotted grid, outward ticks, bordered legends, panel
labels inside the axes, and a muted palette.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.2,
    "axes.labelsize": 9.8,
    "axes.titlesize": 9.8,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.6,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.color": "#c8c8c8",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "legend.borderpad": 0.35,
    "legend.labelspacing": 0.28,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.35,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.2,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

# muted, print-safe palette
C = {
    "blue": "#1f4e79",
    "red": "#c0504d",
    "green": "#4f6228",
    "purple": "#7030a0",
    "orange": "#e08214",
    "teal": "#2e8b8b",
    "grey": "#7f7f7f",
    "lightblue": "#8faadc",
    "lightgreen": "#9dc48a",
    "lightgrey": "#d9d9d9",
    "sand": "#f2dfa8",
    "fillblue": "#dbe5f1",
    "fillpurple": "#e8dfec",
    "fillgreen": "#e2efda",
    "fillsand": "#fff2cc",
    "fillgrey": "#ededed",
}

DET_COLOR = {"yolo11n_vis": C["blue"], "yolo26n_vis": C["red"],
             "rtdetr_l_vis": C["green"], "agnostic": C["purple"],
             "agn_split": C["orange"]}
DET_MARKER = {"yolo11n_vis": "o", "yolo26n_vis": "s", "rtdetr_l_vis": "^",
              "agnostic": "D", "agn_split": "v"}
DET_LABEL = {"yolo11n_vis": "YOLO11n", "yolo26n_vis": "YOLO26n",
             "rtdetr_l_vis": "RT-DETR-l", "agnostic": "agnostic",
             "agn_split": "clean split"}

LABEL = {"mean": "mean fusion", "activity_only": "activity-only",
         "tophat": "top-hat prior", "probe": "learned probe",
         "tophat_permuted": "top-hat, permuted", "probe_permuted": "probe, permuted",
         "tophat_wshuffle": "top-hat, w-shuffled",
         "probe_wshuffle": "probe, w-shuffled", "ensemble": "ensemble",
         "ifcnn": "IFCNN", "densefuse": "DenseFuse", "csf": "classif. saliency",
         "piafusion": "PIAFusion", "spgfusion": "SPGFusion", "vis": "VIS source",
         "ir": "IR source"}


def use():
    plt.rcParams.update(STYLE)


def tidy(ax, grid="y", box=True):
    """Full box frame and a light dotted grid, as in the reference manuscript."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(box)
        ax.spines[side].set_linewidth(0.8)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, axis=grid)
        if grid == "both":
            ax.grid(True, axis="both")
    return ax


def panel(ax, letter, x=0.0, y=1.035, text=None):
    """Bold panel label above the axes, top left, after the reference layout."""
    ax.text(x, y, "(%s)%s" % (letter, " " + text if text else ""),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2,
            fontweight="bold", zorder=6)


def legend(ax, **kw):
    opts = dict(frameon=True, edgecolor="black", fancybox=False, framealpha=1.0,
                borderpad=0.35, labelspacing=0.28, handlelength=1.5,
                handletextpad=0.5, fontsize=8.0)
    opts.update(kw)
    return ax.legend(**opts)


def legend_above(ax, ncol=3, y=1.02, fontsize=8.0, **kw):
    """A bordered legend placed above the axes, so that it never covers data."""
    opts = dict(frameon=True, edgecolor="black", fancybox=False, framealpha=1.0,
                borderpad=0.30, labelspacing=0.25, handlelength=1.4,
                handletextpad=0.45, fontsize=fontsize, ncol=ncol,
                loc="lower center", bbox_to_anchor=(0.5, y), columnspacing=1.0)
    opts.update(kw)
    return ax.legend(**opts)


def interval(ax, y, x, lo, hi, color, marker="o", label=None, ms=4.2, lw=1.0,
             offset=0.0, capsize=2.2):
    """Horizontal interval with a marker at the point estimate."""
    ax.errorbar(x, [v + offset for v in y],
                xerr=[np.array(x, float) - np.array(lo, float),
                      np.array(hi, float) - np.array(x, float)],
                fmt=marker, color=color, ecolor=color, elinewidth=lw,
                capsize=capsize, capthick=lw, ms=ms, lw=0, label=label, zorder=3)
