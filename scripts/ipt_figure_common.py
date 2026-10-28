"""Shared publication-style helpers for the IPT CS-ECF figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULTS = ROOT / "results/selective_fusion"
FIGURE_ROOT = ROOT / "paper/figures_cs_ecf_ipt"

COLORS = {
    "visible": "#68737D",
    "cdf": "#3D6F9E",
    "pia": "#2A9D8F",
    "spg": "#8B5EAA",
    "no_shield": "#D98B2B",
    "csecf": "#174F86",
    "gain": "#2B6EA8",
    "loss": "#C45D4C",
    "neutral": "#C9D1D9",
    "dark": "#263238",
    "ink": "#1F2D3A",
    "mid": "#6B7785",
    "line": "#AAB4BF",
    "light": "#EEF3F7",
    "panel": "#F7F9FB",
    "llvip": "#557FA8",
    "m3fd": "#3F8E89",
    "drone": "#B98345",
}

MARKERS = {
    "always_cdf": "o",
    "always_pia": "^",
    "always_spg": "s",
    "no_shield": "D",
    "cs_ecf": "*",
}


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.edgecolor": COLORS["line"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["mid"],
            "ytick.color": COLORS["mid"],
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "legend.fontsize": 6.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": True,
        }
    )


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def add_panel_label(ax, label, x=-0.08, y=1.03, fontsize=8.2):
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["dark"],
    )


def save_figure(fig, stem, dpi=600):
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("svg", "pdf", "tiff", "png"):
        path = FIGURE_ROOT / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=dpi if suffix in {"tiff", "png"} else None,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
        )
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_csv(filename, fieldnames, rows):
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURE_ROOT / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def box(
    ax,
    xy,
    width,
    height,
    text,
    facecolor,
    edgecolor="#FFFFFF",
    fontsize=7,
    textcolor="#1A1A1A",
    round_radius=0.025,
):
    patch = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.01,rounding_size={round_radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        wrap=True,
    )
    return patch


def arrow(ax, start, end, color="#606060", linewidth=0.9):
    return arrow_patch(ax, start, end, color=color, linewidth=linewidth)


def arrow_patch(
    ax,
    start,
    end,
    color="#606060",
    linewidth=0.9,
    mutation_scale=8,
    connectionstyle="arc3,rad=0",
    linestyle="-",
    zorder=3,
):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": linewidth,
            "linestyle": linestyle,
            "shrinkA": 2,
            "shrinkB": 2,
            "mutation_scale": mutation_scale,
            "connectionstyle": connectionstyle,
        },
        zorder=zorder,
    )


def card(
    ax,
    x,
    y,
    width,
    height,
    title,
    subtitle=None,
    facecolor="#FFFFFF",
    edgecolor=None,
    accent=None,
    titlecolor=None,
    subtitlecolor=None,
    title_size=7.0,
    subtitle_size=6.2,
    shadow=False,
    zorder=3,
):
    edgecolor = edgecolor or COLORS["line"]
    titlecolor = titlecolor or COLORS["ink"]
    subtitlecolor = subtitlecolor or COLORS["mid"]
    if shadow:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x + 0.25, y - 0.25),
                width,
                height,
                boxstyle="round,pad=0.01,rounding_size=0.65",
                facecolor="#263238",
                edgecolor="none",
                alpha=0.07,
                zorder=zorder - 1,
            )
        )
    patch = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.65",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.7,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if accent is not None:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y + height * 0.1),
                0.55,
                height * 0.8,
                boxstyle="round,pad=0.01,rounding_size=0.25",
                facecolor=accent,
                edgecolor="none",
                zorder=zorder + 1,
            )
        )
    text_x = x + width / 2
    if subtitle is None:
        ax.text(
            text_x,
            y + height / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="semibold",
            color=titlecolor,
            zorder=zorder + 2,
        )
    else:
        ax.text(
            text_x,
            y + height * 0.58,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="semibold",
            color=titlecolor,
            zorder=zorder + 2,
        )
        ax.text(
            text_x,
            y + height * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=subtitlecolor,
            linespacing=1.05,
            zorder=zorder + 2,
        )
    return patch


def clean_axis(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["line"])
        ax.spines[side].set_linewidth(0.7)
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color="#E7ECF1",
            linewidth=0.55,
            zorder=0,
        )
        ax.set_axisbelow(True)


def outlined_text(text_artist, width=1.5, foreground="white"):
    text_artist.set_path_effects(
        [withStroke(linewidth=width, foreground=foreground)]
    )
    return text_artist


def marker_handle(label, color, marker="o", markersize=5):
    return Line2D(
        [0],
        [0],
        marker=marker,
        color="none",
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=markersize,
        label=label,
    )

