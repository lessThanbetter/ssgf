"""Shared vector primitives for the manuscript framework figure."""

from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image


BLACK = "#222222"


def arrow(ax, start, end, color="#555555", rad=0.0, lw=0.75, scale=6.5):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1,
            shrinkB=1,
            zorder=4,
        )
    )


def module(
    ax,
    x,
    y,
    width,
    height,
    title,
    subtitle=None,
    face="#FFFFFF",
    edge="#555555",
    title_color=BLACK,
    title_size=7.0,
    subtitle_size=5.8,
    subtitle_dx=0.0,
    subtitle_dy=0.0,
    linewidth=0.7,
):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.15,rounding_size=0.8",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=3,
        )
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.61,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            zorder=5,
        )
        ax.text(
            x + width / 2 + subtitle_dx,
            y + height * 0.31 + subtitle_dy,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=BLACK,
            zorder=5,
        )
    else:
        ax.text(
            x + width / 2,
            y + height / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            zorder=5,
        )


def feature_stack(
    ax,
    x,
    y,
    width,
    height,
    color,
    edge,
    label,
    channels="$N$",
    label_color=BLACK,
):
    for index in range(2, -1, -1):
        offset = index * 0.55
        ax.add_patch(
            Polygon(
                [
                    (x + offset, y + offset),
                    (x + width + offset, y + offset),
                    (x + width + offset + 1.25, y + height + offset),
                    (x + offset + 1.25, y + height + offset),
                ],
                closed=True,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.65,
                alpha=0.55 if index else 1.0,
                zorder=2,
            )
        )
    ax.text(
        x + width / 2 + 0.65,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=label_color,
        zorder=5,
    )
    ax.text(
        x + width / 2 + 0.65,
        y - 1.2,
        channels,
        ha="center",
        va="top",
        fontsize=5.3,
        color="#555555",
        zorder=5,
    )


def image_tile(ax, path: Path, crop, x, y, width, height, label):
    with Image.open(path) as image:
        array = np.asarray(image.crop(crop)).copy()
    ax.imshow(
        array,
        extent=(x, x + width, y, y + height),
        interpolation="lanczos",
        aspect="auto",
        cmap="gray" if array.ndim == 2 else None,
        vmin=0 if array.ndim == 2 else None,
        vmax=255 if array.ndim == 2 else None,
        zorder=2,
    )
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="none",
            edgecolor="#555555",
            linewidth=0.55,
            zorder=3,
        )
    )
    ax.text(
        x + width / 2,
        y - 1.35,
        label,
        ha="center",
        va="top",
        fontsize=6.1,
        color=BLACK,
        zorder=4,
    )
