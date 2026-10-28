"""Figure 4: class concentration and domain-dependent thermal evidence."""

from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ipt_figure_common import (
    COLORS,
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


CLASS_ORDER = ["people", "car", "bus", "motorcycle", "lamp", "truck"]
MODEL_LABELS = {
    "yolo26_m3fd6": "YOLO26n",
    "yolo8n_m3fd6": "YOLOv8n",
}
MODEL_COLORS = {
    "yolo26_m3fd6": "#356B9A",
    "yolo8n_m3fd6": "#C07A57",
}
ASSOCIATIONS = [
    ("thermal_rho", "thermal_p", "Thermal\ntop-hat"),
    ("visible_rho", "visible_p", "Visible\nluminance"),
    ("thermal_mask_rho", "thermal_mask_p", "Mask\nactivity"),
]
DATASETS = [
    ("llvip", "LLVIP"),
    ("m3fd", "M3FD"),
    ("dronevehicle_test", "DroneVehicle"),
]


def _stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def main() -> None:
    apply_style()
    class_data = load_json("confidence_shield_class_bootstrap.json")["m3fd"]
    counts = load_json("m3fd_class_summary.json")["annotations"]
    mask_data = load_json("ipt_mask_ablation.json")
    thermal_data = load_json("ipt_thermal_stratification.json")
    source_rows = []

    fig = plt.figure(figsize=(6.35, 6.1))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.52, 1.0],
        height_ratios=[1.25, 1.0],
        hspace=0.52,
        wspace=0.40,
        left=0.12,
        right=0.97,
        top=0.93,
        bottom=0.08,
    )

    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(len(CLASS_ORDER))
    offsets = {"yolo26_m3fd6": -0.15, "yolo8n_m3fd6": 0.15}
    ax.axhspan(-0.46, 0.46, color="#EFF5FA", zorder=0)
    for model, label in MODEL_LABELS.items():
        gains = []
        lows = []
        highs = []
        for class_id in range(len(CLASS_ORDER)):
            row = class_data[model][str(class_id)]
            gain = 100 * row["gain"]
            low, high = [100 * value for value in row["image_ci95"]]
            gains.append(gain)
            lows.append(gain - low)
            highs.append(max(0.0, high - gain))
            source_rows.append(
                {
                    "panel": "a",
                    "model": label,
                    "class": CLASS_ORDER[class_id],
                    "gain_ap50_points": round(gain, 6),
                    "ci_low_ap50_points": round(low, 6),
                    "ci_high_ap50_points": round(high, 6),
                    "annotations": counts[CLASS_ORDER[class_id]],
                }
            )
        y_pos = y + offsets[model]
        ax.errorbar(
            gains,
            y_pos,
            xerr=np.asarray([lows, highs]),
            fmt="o",
            markersize=4.1,
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
            zorder=4,
        )

    ax.axvline(0, color=COLORS["ink"], linewidth=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{name.capitalize()}  (n={counts[name]})"
            for name in CLASS_ORDER
        ],
        fontsize=5.9,
    )
    ax.invert_yaxis()
    ax.set_xlim(-1.2, 20.5)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("M3FD AP50 change (percentage points)", fontsize=6.2)
    ax.set_title("Class concentration in M3FD", loc="left", pad=18, fontsize=7.0)
    clean_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.legend(
        handles=[
            marker_handle(label, MODEL_COLORS[model], "o", 4.2)
            for model, label in MODEL_LABELS.items()
        ],
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=5.8,
    )
    add_panel_label(ax, "a", x=-0.17, y=1.04)

    ax2 = fig.add_subplot(grid[0, 1])
    checkpoints = ["best", "last"]
    mask_values = [100 * mask_data[checkpoint]["thermal_mask_gain"] for checkpoint in checkpoints]
    global_values = [100 * mask_data[checkpoint]["global_gain"] for checkpoint in checkpoints]
    y2 = np.arange(2)
    for index, checkpoint in enumerate(checkpoints):
        ax2.plot(
            [global_values[index], mask_values[index]],
            [y2[index], y2[index]],
            color=COLORS["line"],
            linewidth=1.0,
            zorder=1,
        )
        ax2.scatter(
            global_values[index],
            y2[index],
            s=36,
            marker="o",
            color=COLORS["neutral"],
            edgecolor=COLORS["mid"],
            linewidth=0.5,
            zorder=3,
        )
        ax2.scatter(
            mask_values[index],
            y2[index],
            s=45,
            marker="D",
            color=COLORS["cdf"],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        ax2.text(
            mask_values[index] + 0.025,
            y2[index],
            f"{mask_values[index]:.2f}",
            ha="left",
            va="center",
            fontsize=5.5,
            color=COLORS["cdf"],
            fontweight="semibold",
        )
        source_rows.append(
            {
                "panel": "b",
                "checkpoint": checkpoint,
                "thermal_mask_gain_ap50_points": round(mask_values[index], 6),
                "global_gain_ap50_points": round(global_values[index], 6),
                "mask_minus_global_points": round(
                    mask_values[index] - global_values[index], 6
                ),
            }
        )
    ax2.axvline(0, color=COLORS["ink"], linewidth=0.75)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(["best", "last"], fontsize=6.1)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, 0.71)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax2.set_xlabel("CDF AP50 gain (points)", fontsize=6.2)
    ax2.set_title("Thermal-mask ablation", loc="left", pad=18, fontsize=7.0)
    clean_axis(ax2, grid_axis="x")
    ax2.tick_params(axis="y", length=0, pad=3)
    ax2.legend(
        handles=[
            marker_handle("Global injection", COLORS["neutral"], "o", 4.0),
            marker_handle("Thermal mask", COLORS["cdf"], "D", 4.0),
        ],
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        handletextpad=0.35,
        fontsize=5.7,
    )
    ax2.text(
        0.69,
        -0.10,
        "Δ +0.03 / +0.12",
        ha="right",
        va="top",
        fontsize=5.2,
        color=COLORS["mid"],
    )
    add_panel_label(ax2, "b", x=-0.28, y=1.04)

    ax3 = fig.add_subplot(grid[1, :])
    matrix = np.asarray(
        [
            [
                thermal_data[dataset]["correlation_with_switch_fraction"][rho_key]
                for rho_key, _, _ in ASSOCIATIONS
            ]
            for dataset, _ in DATASETS
        ]
    )
    p_values = np.asarray(
        [
            [
                thermal_data[dataset]["correlation_with_switch_fraction"][p_key]
                for _, p_key, _ in ASSOCIATIONS
            ]
            for dataset, _ in DATASETS
        ]
    )
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "thermal_assoc",
        ["#A64F45", "#D98A79", "#F4E9E7", "#FAFAF8", "#DCECF3", "#79A8C4", "#2D678F"],
        N=256,
    )
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    image = ax3.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax3.set_xticks(np.arange(len(ASSOCIATIONS)))
    ax3.set_xticklabels([label for _, _, label in ASSOCIATIONS], fontsize=6.0)
    ax3.set_yticks(np.arange(len(DATASETS)))
    ax3.set_yticklabels([label for _, label in DATASETS], fontsize=6.0)
    ax3.tick_params(length=0, pad=4)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.set_xticks(np.arange(-0.5, len(ASSOCIATIONS), 1), minor=True)
    ax3.set_yticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax3.grid(which="minor", color="white", linewidth=1.1)
    ax3.tick_params(which="minor", bottom=False, left=False)

    for row_index, (dataset, display) in enumerate(DATASETS):
        for col_index, (rho_key, p_key, _) in enumerate(ASSOCIATIONS):
            rho = matrix[row_index, col_index]
            p_value = p_values[row_index, col_index]
            red, green, blue, _ = cmap(norm(rho))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            text_color = "white" if luminance < 0.45 else COLORS["ink"]
            ax3.text(
                col_index,
                row_index,
                f"{rho:+.2f}{_stars(float(p_value))}",
                ha="center",
                va="center",
                fontsize=6.1,
                color=text_color,
                fontweight="semibold",
            )
            source_rows.append(
                {
                    "panel": "c",
                    "dataset": display,
                    "association": dict(
                        (key, label.replace("\n", " "))
                        for key, _, label in ASSOCIATIONS
                    )[rho_key],
                    "spearman_rho": round(float(rho), 6),
                    "p_value": float(p_value),
                }
            )

    ax3.set_title("Thermal association reverses across domains", loc="left", pad=9, fontsize=7.0)
    ax3.set_xlabel(
        "Spearman ρ with switched fraction; *p<0.05, **p<0.01, ***p<0.001",
        fontsize=5.8,
        labelpad=5,
    )
    cbar = fig.colorbar(
        image,
        ax=ax3,
        fraction=0.026,
        pad=0.018,
        ticks=[-0.5, -0.25, 0, 0.25, 0.5],
    )
    cbar.ax.tick_params(labelsize=5.4, length=2)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label("Spearman ρ", fontsize=5.8, labelpad=3)
    add_panel_label(ax3, "c", x=-0.13, y=1.06)

    write_csv(
        "figure4_source.csv",
        [
            "panel",
            "model",
            "class",
            "gain_ap50_points",
            "ci_low_ap50_points",
            "ci_high_ap50_points",
            "annotations",
            "checkpoint",
            "thermal_mask_gain_ap50_points",
            "global_gain_ap50_points",
            "mask_minus_global_points",
            "dataset",
            "association",
            "spearman_rho",
            "p_value",
        ],
        source_rows,
    )
    save_figure(fig, "fig4_thermal_class_boundary", dpi=600)


if __name__ == "__main__":
    main()
