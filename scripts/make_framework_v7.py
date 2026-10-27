"""Redraw Figure 1 while retaining the original stage-based visual language."""

# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# architecture schematic -> assets/figures/other -> cross-type inherit, parameter inherit
# The established Figure 1 palette and stage grammar are preserved from framework_v5.

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle

from framework_figure_utils import arrow, feature_stack, image_tile, module


# Preserve the original Figure 1 style: STIX/Times-family serif with math italics.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


ROOT = Path(__file__).resolve().parents[1]


def draw_boxes(ax, annotation: Path, crop, extent, color="#F1A340"):
    if not annotation.is_file():
        return
    crop_x1, crop_y1, crop_x2, crop_y2 = crop
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    x, y, width, height = extent
    image_width, image_height = 1280, 1024
    for line in annotation.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        _, center_x, center_y, box_width, box_height = map(float, values)
        x1 = (center_x - box_width / 2) * image_width
        x2 = (center_x + box_width / 2) * image_width
        y1 = (center_y - box_height / 2) * image_height
        y2 = (center_y + box_height / 2) * image_height
        x1 = max(x1, crop_x1)
        x2 = min(x2, crop_x2)
        y1 = max(y1, crop_y1)
        y2 = min(y2, crop_y2)
        if x2 <= x1 or y2 <= y1:
            continue
        left = x + (x1 - crop_x1) / crop_width * width
        right = x + (x2 - crop_x1) / crop_width * width
        bottom = y + (crop_y2 - y2) / crop_height * height
        top = y + (crop_y2 - y1) / crop_height * height
        ax.add_patch(
            Rectangle(
                (left, bottom),
                right - left,
                top - bottom,
                facecolor="none",
                edgecolor=color,
                linewidth=0.55,
                zorder=4,
            )
        )


def draw_kernel(ax, x, y, size):
    cells = 5
    step = size / cells
    for index in range(cells + 1):
        ax.plot(
            [x + index * step, x + index * step],
            [y, y + size],
            color="#6D8AA8",
            linewidth=0.35,
            zorder=4,
        )
        ax.plot(
            [x, x + size],
            [y + index * step, y + index * step],
            color="#6D8AA8",
            linewidth=0.35,
            zorder=4,
        )


def save(fig, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    base = output / "framework_v7"
    fig.savefig(f"{base}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base}.svg", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return base


def main(source_root: Path, output: Path) -> None:
    mm = 1 / 25.4
    fig, ax = plt.subplots(figsize=(183 * mm, 100 * mm))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ink = "#222222"
    blue = "#2166AC"
    blue_fill = "#DCEAF7"
    blue_edge = "#8FB2D9"
    orange = "#F1A340"
    orange_fill = "#FDB786"
    orange_edge = "#DA995F"
    purple = "#762A83"
    purple_fill = "#EADFF0"
    green = "#1B7837"
    green_fill = "#CCE7BA"
    green_edge = "#8CB66E"
    crop = (35, 455, 990, 1010)

    # Original two-stage composition, redrawn with cleaner spacing.
    ax.add_patch(
        FancyBboxPatch(
            (1.0, 45.0),
            128.0,
            51.5,
            boxstyle="round,pad=0.25,rounding_size=1.4",
            facecolor="#EEF5FB",
            edgecolor=blue_edge,
            linewidth=0.9,
            zorder=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (1.0, 2.5),
            128.0,
            36.5,
            boxstyle="round,pad=0.25,rounding_size=1.4",
            facecolor="#FFF5E9",
            edgecolor="#DDB77A",
            linewidth=0.9,
            zorder=0,
        )
    )

    ax.text(
        65.0,
        94.9,
        "STAGE 1: PER-SOURCE TARGETNESS WEIGHTING",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=ink,
    )
    ax.text(
        65.0,
        92.1,
        r"shared features $\rightarrow$ targetness $\times$ activity $\rightarrow$ softmax weights",
        ha="center",
        va="center",
        fontsize=5.8,
        fontstyle="italic",
        color="#444444",
    )
    ax.text(
        65.0,
        37.1,
        "STAGE 2: WEIGHTED FUSION AND RECONSTRUCTION",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=ink,
    )
    ax.text(
        65.0,
        34.3,
        r"temperature softmax ($\tau=0.3$), convex feature combination and mirrored decoding",
        ha="center",
        va="center",
        fontsize=5.7,
        fontstyle="italic",
        color="#444444",
    )

    image_tile(
        ax,
        source_root / "test/visible/190003.jpg",
        crop,
        4.2,
        78.0,
        10.7,
        6.3,
        "Visible $I_v$",
    )
    image_tile(
        ax,
        source_root / "test/infrared/190003.jpg",
        crop,
        4.2,
        58.2,
        10.7,
        6.3,
        "Infrared $I_r$",
    )
    annotation = source_root / "test/labels/190003.txt"
    draw_boxes(
        ax,
        annotation,
        crop,
        (4.2, 78.0, 10.7, 6.3),
    )
    draw_boxes(
        ax,
        annotation,
        crop,
        (4.2, 58.2, 10.7, 6.3),
    )

    module(
        ax,
        18.6,
        62.7,
        8.8,
        17.0,
        "Encoder",
        "$E$, shared",
        face="#FFFFFF",
        edge="#666666",
        title_size=6.8,
        subtitle_size=5.4,
        subtitle_dx=-0.270,
        subtitle_dy=0.619,
    )
    draw_kernel(ax, 20.03, 64.448, 2.3)
    ax.text(23.10, 64.978, r"$5\times5$", ha="left", va="center",
            fontsize=5.0, color="#4E6D8A")
    arrow(ax, (15.1, 81.1), (18.4, 76.0), rad=-0.08)
    arrow(ax, (15.1, 61.3), (18.4, 66.3), rad=0.08)

    feature_stack(
        ax,
        31.8,
        76.2,
        8.2,
        8.3,
        orange_fill,
        orange_edge,
        "$F_v$",
        "24 channels",
    )
    feature_stack(
        ax,
        31.8,
        56.0,
        8.2,
        8.3,
        "#BACEEF",
        "#6C93C4",
        "$F_r$",
        "24 channels",
    )
    arrow(ax, (27.7, 76.0), (31.6, 80.3), rad=-0.08)
    arrow(ax, (27.7, 66.3), (31.6, 60.1), rad=0.08)

    ax.add_patch(
        FancyBboxPatch(
            (45.0, 51.7),
            31.8,
            36.3,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            facecolor=purple_fill,
            edgecolor=purple,
            linewidth=0.7,
            alpha=0.72,
            zorder=1,
        )
    )
    ax.text(
        60.9,
        85.5,
        "Targetness weighting",
        ha="center",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=purple,
    )

    for source, y, color in (("v", 75.8, orange), ("r", 57.8, blue)):
        module(
            ax,
            47.3,
            y,
            6.0,
            7.2,
            f"$A_{source}$",
            face="#FFFFFF",
            edge=color,
            title_size=6.5,
        )
        module(
            ax,
            55.1,
            y,
            6.0,
            7.2,
            f"$S_{source}$",
            face="#FFFFFF",
            edge=purple,
            title_size=6.5,
        )
        module(
            ax,
            63.0,
            y,
            11.1,
            7.2,
            rf"$d_{source}=S_{source}\odot A_{source}$",
            face="#FFFFFF",
            edge="#777777",
            title_size=5.8,
        )
        ax.plot(
            [53.4, 54.9],
            [y + 3.6, y + 3.6],
            color="#777777",
            linewidth=0.55,
            zorder=4,
        )
        ax.text(
            54.15,
            y + 3.6 - (0.248 if source == "v" else 0.124),
            r"$\odot$",
            ha="center",
            va="center",
            fontsize=6.2,
            color=purple,
            zorder=5,
        )
        arrow(ax, (61.2, y + 3.6), (62.8, y + 3.6), scale=5.4)

    arrow(ax, (40.6, 80.3), (47.1, 79.4), scale=5.4)
    arrow(ax, (40.6, 60.1), (47.1, 61.4), scale=5.4)
    ax.text(
        60.9,
        53.5,
        r"$S_s\in\{\mathrm{top\!-\!hat},\ \Phi\}$",
        ha="center",
        va="center",
        fontsize=5.7,
        color="#444444",
    )

    module(
        ax,
        81.3,
        62.5,
        10.4,
        15.8,
        "softmax",
        r"$w_s(d_s/m\tau)$",
        face="#FFFFFF",
        edge=purple,
        title_size=6.6,
        subtitle_size=5.3,
    )
    module(
        ax,
        96.0,
        62.0,
        11.0,
        16.8,
        "Weights",
        r"$w_v,\ w_r$",
        face="#FFFFFF",
        edge=purple,
        title_size=6.5,
        subtitle_size=6.2,
    )

    map_data = np.load(ROOT / "paper" / "framework_v7_maps.npz")
    target_cmap = LinearSegmentedColormap.from_list(
        "targetness",
        ["#F7F7F7", "#D7BFE4", "#762A83"],
    )
    weight_cmap = LinearSegmentedColormap.from_list(
        "modality_weight",
        ["#2166AC", "#F7F7F7", "#F1A340"],
    )
    map_x = 109.5
    map_w = 16.8
    ax.text(
        map_x + map_w / 2,
        80.893,
        r"probe targetness $S_v$",
        ha="center",
        va="bottom",
        fontsize=5.3,
        color="#555555",
    )
    ax.imshow(
        map_data["s_vis"],
        extent=(map_x, map_x + map_w, 72.793, 80.093),
        cmap=target_cmap,
        norm=Normalize(vmin=0, vmax=1),
        interpolation="bilinear",
        aspect="auto",
        zorder=2,
    )
    ax.add_patch(
        Rectangle(
            (map_x, 72.793),
            map_w,
            7.3,
            facecolor="none",
            edgecolor="#777777",
            linewidth=0.45,
            zorder=3,
        )
    )
    ax.text(
        map_x + map_w / 2,
        70.593,
        r"normalized visible weight $w_v$",
        ha="center",
        va="bottom",
        fontsize=5.3,
        color="#555555",
    )
    ax.imshow(
        map_data["w_vis"],
        extent=(map_x, map_x + map_w, 62.193, 69.693),
        cmap=weight_cmap,
        norm=TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1),
        interpolation="bilinear",
        aspect="auto",
        zorder=2,
    )
    ax.add_patch(
        Rectangle(
            (map_x, 62.193),
            map_w,
            7.5,
            facecolor="none",
            edgecolor="#777777",
            linewidth=0.45,
            zorder=3,
        )
    )
    ax.text(
        map_x + 0.25,
        61.393,
        "IR",
        ha="left",
        va="top",
        fontsize=5.0,
        color=blue,
    )
    ax.text(
        map_x + map_w - 0.25,
        61.393,
        "VIS",
        ha="right",
        va="top",
        fontsize=5.0,
        color=orange,
    )
    arrow(ax, (77.0, 80.0), (81.1, 74.8), rad=-0.08, color=purple)
    arrow(ax, (77.0, 61.4), (81.1, 66.2), rad=0.08, color=purple)
    arrow(ax, (91.9, 70.4), (95.8, 70.4), color=purple)
    ax.text(87.0, 59.2, r"$\tau=0.3$", ha="center", va="center",
            fontsize=5.4, color="#555555")

    # Explicit hand-off from per-source weights to the reconstruction stage.
    connector = "#D4943D"
    ax.plot([101.5, 101.5], [61.9, 42.2], color=connector, linewidth=1.1, zorder=2)
    ax.plot([101.5, 13.0], [42.2, 42.2], color=connector, linewidth=1.1, zorder=2)
    arrow(ax, (13.0, 42.2), (13.0, 29.0), color=connector, lw=1.1, scale=7.5)
    ax.text(
        58.439,
        42.132,
        r"per-source weights $w_v,\ w_r$",
        ha="center",
        va="center",
        fontsize=5.8,
        color=connector,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
        zorder=5,
    )

    # Stage 2: weighted feature fusion and reconstruction.
    module(
        ax,
        6.5,
        13.0,
        12.5,
        15.8,
        "Weights",
        r"$w_v,\ w_r$",
        face="#FFFFFF",
        edge=purple,
        title_size=6.0,
        subtitle_size=5.4,
    )
    module(
        ax,
        25.0,
        10.8,
        25.8,
        19.3,
        "Weighted fusion",
        r"$\hat F=\sum_s w_s\odot F_s$",
        face="#FFFFFF",
        edge=green,
        title_size=6.4,
        subtitle_size=6.1,
    )
    feature_stack(
        ax,
        57.5,
        14.5,
        10.0,
        9.0,
        green_fill,
        green_edge,
        r"$\hat F$",
        "24 channels",
    )
    module(
        ax,
        76.0,
        12.5,
        14.0,
        17.2,
        "Decoder",
        "$D$, mirrored",
        face="#FFFFFF",
        edge=green,
        title_size=6.5,
        subtitle_size=5.3,
    )
    image_tile(
        ax,
        source_root / "k1_native/imgs/probe/190003.png",
        crop,
        101.0,
        14.0,
        18.0,
        10.7,
        "Fused $I_f$",
    )
    draw_boxes(
        ax,
        annotation,
        crop,
        (101.0, 14.0, 18.0, 10.7),
    )
    arrow(ax, (19.1, 20.9), (24.8, 20.9), color=purple)
    arrow(ax, (50.9, 20.5), (57.3, 20.0), color=green)
    arrow(ax, (68.1, 20.0), (75.8, 20.0), color=green)
    arrow(ax, (90.1, 20.0), (100.8, 20.0), color=green)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    print(save(fig, output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=r"D:\Code\_ssgf_external\r2_clean",
        help="Directory containing LLVIP test and probe-fusion images",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "paper"),
        help="Output directory",
    )
    options = parser.parse_args()
    main(Path(options.source_root), Path(options.output))
