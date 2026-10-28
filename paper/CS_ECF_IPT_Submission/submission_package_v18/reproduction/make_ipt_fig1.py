"""Figure 1: evaluator-conditioned utility and dataset-level sign reversals."""

from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import patches

from ipt_figure_common import (
    COLORS,
    MARKERS,
    add_panel_label,
    apply_style,
    clean_axis,
    load_json,
    marker_handle,
    save_figure,
    write_csv,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
PUBLICATION_EXPORTS = (".svg", ".pdf", ".tiff", ".png")


CHECKPOINTS = [
    ("llvip", "yolo11", "YOLO11n"),
    ("llvip", "agnostic", "Modality-agnostic"),
    ("llvip", "yolo26_s0", "YOLO26n s0"),
    ("llvip", "rtdetr", "RT-DETR-l"),
    ("llvip", "yolo26_s1", "YOLO26n s1"),
    ("llvip", "yolo26_s2", "YOLO26n s2"),
    ("llvip", "yolo8_s0", "YOLOv8n s0"),
    ("llvip", "yolo8_s1", "YOLOv8n s1"),
    ("m3fd", "yolo26_m3fd6", "YOLO26n"),
    ("m3fd", "yolo8n_m3fd6", "YOLOv8n"),
    ("dronevehicle_test", "best", "OBB best"),
    ("dronevehicle_test", "last", "OBB last"),
]
POLICIES = [
    ("always_cdf", "Always CDF", COLORS["cdf"]),
    ("always_pia", "Always PIAFusion", COLORS["pia"]),
    ("always_spg", "Always SPGFusion", COLORS["spg"]),
    ("no_shield", "No shield", COLORS["no_shield"]),
    ("cs_ecf", "CS-ECF", COLORS["csecf"]),
]
DATASETS = [
    ("llvip", "LLVIP", COLORS["llvip"]),
    ("m3fd", "M3FD", COLORS["m3fd"]),
    ("dronevehicle_test", "DroneVehicle", COLORS["drone"]),
]


def _cell_text_color(image, value):
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.47 else COLORS["ink"]


def main() -> None:
    apply_style()
    data = load_json("confirmatory_policy_comparison.json")

    matrix = []
    source_rows = []
    for dataset, checkpoint, display in CHECKPOINTS:
        row = data["per_checkpoint"][f"{dataset}/{checkpoint}"]
        matrix.append([100 * row[policy] for policy, _, _ in POLICIES])
        for policy, display_policy, _ in POLICIES:
            source_rows.append(
                {
                    "dataset": dataset,
                    "checkpoint": checkpoint,
                    "display_checkpoint": display,
                    "policy": display_policy,
                    "gain_ap50_points": round(100 * row[policy], 6),
                }
            )
    matrix = np.asarray(matrix)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ipt_signed_gain",
        [
            "#A93F3F",
            "#D87364",
            "#F1B7AC",
            "#FAF8F5",
            "#B7D3E8",
            "#5E9BC4",
            "#1E5D8F",
        ],
        N=256,
    )
    limit = max(abs(matrix.min()), abs(matrix.max()))

    fig = plt.figure(figsize=(6.35, 5.8))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.72, 1.12],
        hspace=0.55,
        left=0.205,
        right=0.93,
        top=0.92,
        bottom=0.10,
    )

    ax = fig.add_subplot(grid[0])
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(POLICIES)))
    ax.set_xticklabels(
        ["Always\nCDF", "Always\nPIAFusion", "Always\nSPGFusion", "No\nshield", "CS-ECF"],
        fontsize=6.5,
        linespacing=0.95,
    )
    ax.set_yticks(np.arange(len(CHECKPOINTS)))
    dataset_short = {
        "llvip": "LLVIP",
        "m3fd": "M3FD",
        "dronevehicle_test": "DV",
    }
    ax.set_yticklabels(
        [
            f"{dataset_short[dataset]} · {display}"
            for dataset, _, display in CHECKPOINTS
        ],
        fontsize=6.2,
    )
    ax.tick_params(axis="both", length=0, pad=3)
    ax.set_xticks(np.arange(-0.5, len(POLICIES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CHECKPOINTS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=6.0,
                color=_cell_text_color(image, value),
                fontweight="semibold" if col_index == 4 else "normal",
            )

    for boundary in (7.5, 9.5):
        ax.axhline(boundary, color="white", linewidth=2.3, zorder=3)
        ax.axhline(boundary, color=COLORS["line"], linewidth=0.3, alpha=0.55, zorder=3)

    ax.add_patch(
        patches.Rectangle(
            (3.5, -0.5),
            1.0,
            len(CHECKPOINTS),
            fill=False,
            edgecolor=COLORS["csecf"],
            linewidth=1.35,
            zorder=5,
        )
    )
    ax.text(
        4,
        -0.92,
        "selected policy",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=COLORS["csecf"],
        fontweight="bold",
        clip_on=False,
    )
    ax.set_title("Signed AP50 change from visible", loc="left", pad=16, fontsize=7.5)
    add_panel_label(ax, "a", x=-0.18, y=1.055)

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.024,
        pad=0.025,
        ticks=[-25, -15, -5, 0, 5, 15, 25],
    )
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label("AP50 change (points)", fontsize=6.2, labelpad=4)

    ax2 = fig.add_subplot(grid[1])
    policy_offsets = np.linspace(-0.24, 0.24, len(POLICIES))
    for dataset_index, (dataset, _, dataset_color) in enumerate(DATASETS):
        ax2.axhspan(
            dataset_index - 0.42,
            dataset_index + 0.42,
            color=dataset_color,
            alpha=0.045 if dataset_index % 2 == 0 else 0.02,
            zorder=0,
        )
        for policy_index, ((policy, label, color), offset) in enumerate(
            zip(POLICIES, policy_offsets)
        ):
            value = 100 * data["summaries"][policy]["profile_means"][dataset]
            y = dataset_index + offset
            ax2.plot(
                [0, value],
                [y, y],
                color=color,
                linewidth=0.65,
                alpha=0.28,
                zorder=1,
            )
            ax2.scatter(
                value,
                y,
                s=60 if policy == "cs_ecf" else 27,
                marker=MARKERS[policy],
                color=color,
                edgecolor="white" if policy != "always_cdf" else COLORS["line"],
                linewidth=0.55,
                zorder=4,
            )
            if policy in {"no_shield", "cs_ecf"}:
                ha = "left" if value >= 0 else "right"
                shift = 0.32 if value >= 0 else -0.32
                ax2.text(
                    value + shift,
                    y,
                    f"{value:+.1f}",
                    ha=ha,
                    va="center",
                    fontsize=5.7,
                    color=color,
                    fontweight="semibold" if policy == "cs_ecf" else "normal",
                )

    ax2.axvline(0, color=COLORS["ink"], linewidth=0.75, zorder=2)
    ax2.set_xlim(-19, 15)
    ax2.set_ylim(len(DATASETS) - 0.45, -0.55)
    ax2.set_yticks(range(len(DATASETS)))
    ax2.set_yticklabels([label for _, label, _ in DATASETS], fontsize=6.6)
    ax2.set_xlabel("Mean AP50 change from visible (points)", fontsize=6.6)
    ax2.set_title("Dataset-level policy means", loc="left", pad=22, fontsize=7.2)
    clean_axis(ax2, grid_axis="x")
    ax2.tick_params(axis="y", length=0, pad=3)
    handles = [marker_handle(label, color, MARKERS[policy]) for policy, label, color in POLICIES]
    ax2.legend(
        handles=handles,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        columnspacing=1.1,
        handletextpad=0.4,
        fontsize=5.8,
    )
    add_panel_label(ax2, "b", x=-0.18, y=1.04)

    write_csv(
        "figure1_source.csv",
        [
            "dataset",
            "checkpoint",
            "display_checkpoint",
            "policy",
            "gain_ap50_points",
        ],
        source_rows,
    )
    save_figure(fig, "fig1_evaluator_dependence", dpi=600)


if __name__ == "__main__":
    main()
