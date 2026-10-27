"""Shared publication-style helpers for the IPT CS-ECF figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402


ROOT = Path(r"D:\Code\_server_handoff_v2\ssgf")
RESULTS = ROOT / "results/selective_fusion"
FIGURE_ROOT = ROOT / "paper/figures_cs_ecf_ipt"

COLORS = {
    "visible": "#606060",
    "cdf": "#3775BA",
    "pia": "#42949E",
    "spg": "#9A4D8E",
    "no_shield": "#E28E2C",
    "csecf": "#0F4D92",
    "gain": "#2E9E44",
    "loss": "#B64342",
    "neutral": "#B8B8B8",
    "dark": "#272727",
    "light": "#E8EDF3",
}


def apply_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.7
    plt.rcParams["ytick.major.width"] = 0.7
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["ytick.major.size"] = 3


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def add_panel_label(ax, label, x=-0.08, y=1.03):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
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
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": linewidth,
            "shrinkA": 2,
            "shrinkB": 2,
            "mutation_scale": 8,
        },
    )

