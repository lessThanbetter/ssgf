"""Figure 1: evaluator-dependent signed AP50 changes."""

from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from ipt_figure_common import (
    COLORS,
    add_panel_label,
    apply_style,
    load_json,
    save_figure,
    write_csv,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
PUBLICATION_EXPORTS = (".svg", ".pdf", ".tiff", ".png")
PUBLICATION_DPI = 600


CHECKPOINTS = [
    ("llvip", "yolo11", "LLVIP / YOLO11n"),
    ("llvip", "agnostic", "LLVIP / Agnostic"),
    ("llvip", "yolo26_s0", "LLVIP / YOLO26n s0"),
    ("llvip", "rtdetr", "LLVIP / RT-DETR-l"),
    ("llvip", "yolo26_s1", "LLVIP / YOLO26n s1"),
    ("llvip", "yolo26_s2", "LLVIP / YOLO26n s2"),
    ("llvip", "yolo8_s0", "LLVIP / YOLOv8n s0"),
    ("llvip", "yolo8_s1", "LLVIP / YOLOv8n s1"),
    ("m3fd", "yolo26_m3fd6", "M3FD / YOLO26n"),
    ("m3fd", "yolo8n_m3fd6", "M3FD / YOLOv8n"),
    ("dronevehicle_test", "best", "DroneVehicle / best"),
    ("dronevehicle_test", "last", "DroneVehicle / last"),
]
POLICIES = [
    ("always_cdf", "CDF"),
    ("always_pia", "PIAFusion"),
    ("always_spg", "SPGFusion"),
    ("no_shield", "No shield"),
    ("cs_ecf", "CS-ECF"),
]
DATASETS = [
    ("llvip", "LLVIP"),
    ("m3fd", "M3FD"),
    ("dronevehicle_test", "DroneVehicle"),
]


def main():
    apply_style()
    data = load_json("confirmatory_policy_comparison.json")
    source_rows = []
    matrix = []
    for dataset, checkpoint, display in CHECKPOINTS:
        row = data["per_checkpoint"][f"{dataset}/{checkpoint}"]
        matrix.append([100 * row[policy] for policy, _ in POLICIES])
        for policy, display_policy in POLICIES:
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

    fig = plt.figure(figsize=(6.25, 6.0))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.35, 1.35],
        hspace=0.38,
    )
    ax = fig.add_subplot(grid[0])
    limit = np.max(np.abs(matrix))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "signed_gain",
        ["#B64342", "#F6CFCB", "#F7F7F7", "#B4C0E4", "#0F4D92"],
    )
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    ax.set_xticks(range(len(POLICIES)))
    ax.set_xticklabels(
        [label for _, label in POLICIES],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(CHECKPOINTS)))
    ax.set_yticklabels([display for _, _, display in CHECKPOINTS])
    ax.tick_params(length=0)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text_color = "white" if abs(value) > 0.13 * limit else "#202020"
            ax.text(
                col_index,
                row_index,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=6.4,
                color=text_color,
                fontweight="bold" if col_index == 4 else "normal",
            )
    ax.axhline(7.5, color="white", linewidth=2.2)
    ax.axhline(9.5, color="white", linewidth=2.2)
    ax.set_title("Signed AP50 change from visible input", loc="left", pad=8)
    add_panel_label(ax, "a", x=-0.24, y=1.02)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("AP50 change (points)")
    cbar.outline.set_linewidth(0.5)

    ax2 = fig.add_subplot(grid[1])
    x = np.arange(len(DATASETS))
    width = 0.15
    policy_colors = [
        COLORS["cdf"],
        COLORS["pia"],
        COLORS["spg"],
        COLORS["no_shield"],
        COLORS["csecf"],
    ]
    hatches = ["", "//", "..", "xx", ""]
    for policy_index, ((policy, policy_label), color, hatch) in enumerate(
        zip(POLICIES, policy_colors, hatches)
    ):
        values = [
            100 * data["summaries"][policy]["profile_means"][dataset]
            for dataset, _ in DATASETS
        ]
        bars = ax2.bar(
            x + (policy_index - 2) * width,
            values,
            width=width,
            color=color,
            edgecolor=COLORS["dark"] if policy in {"always_cdf", "cs_ecf"} else "white",
            linewidth=0.6,
            label=policy_label,
        )
        for bar in bars:
            bar.set_hatch(hatch)
    ax2.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax2.set_ylim(-18, 25)
    ax2.set_xticks(x)
    ax2.set_xticklabels([label for _, label in DATASETS])
    ax2.set_ylabel("Mean AP50 change (points)")
    ax2.legend(ncol=3, loc="upper right", fontsize=6.2)
    add_panel_label(ax2, "b", x=-0.12, y=1.03)
    ax2.set_title("Dataset-level mean sign reversals", loc="left", pad=6)

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
    save_figure(fig, "fig1_evaluator_dependence", dpi=PUBLICATION_DPI)


if __name__ == "__main__":
    main()
