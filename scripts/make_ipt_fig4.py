"""Figure 4: thermal evidence and class boundary."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
PUBLICATION_EXPORTS = (".svg", ".pdf", ".tiff", ".png")
PUBLICATION_DPI = 600


CLASS_ORDER = ["people", "car", "bus", "motorcycle", "lamp", "truck"]
MODEL_LABELS = {
    "yolo26_m3fd6": "YOLO26n",
    "yolo8n_m3fd6": "YOLOv8n",
}


def main():
    apply_style()
    class_data = load_json("confidence_shield_class_bootstrap.json")["m3fd"]
    mask_data = load_json("ipt_mask_ablation.json")
    thermal_data = load_json("ipt_thermal_stratification.json")
    source_rows = []

    fig = plt.figure(figsize=(7.2, 6.0))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.3, 1.0],
        height_ratios=[1.35, 0.85],
        hspace=0.55,
        wspace=0.38,
    )

    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(CLASS_ORDER))
    width = 0.34
    for index, (model, label) in enumerate(MODEL_LABELS.items()):
        values = [
            100 * class_data[model][str(class_id)]["gain"]
            if str(class_id) in class_data[model]
            else 0
            for class_id in range(6)
        ]
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width=width,
            color=COLORS["cdf"] if index == 0 else COLORS["pia"],
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
        for class_id, value in enumerate(values):
            source_rows.append(
                {
                    "panel": "a",
                    "model": label,
                    "class": CLASS_ORDER[class_id],
                    "gain_ap50_points": round(value, 6),
                }
            )
    ax.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [name.capitalize() for name in CLASS_ORDER],
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("AP50 change (points)")
    ax.set_title("M3FD class concentration", loc="left", pad=7)
    ax.legend(loc="upper right")
    add_panel_label(ax, "a", x=-0.18, y=1.02)

    ax2 = fig.add_subplot(grid[0, 1])
    checkpoints = ["best", "last"]
    mask_values = [
        100 * mask_data[checkpoint]["thermal_mask_gain"]
        for checkpoint in checkpoints
    ]
    global_values = [
        100 * mask_data[checkpoint]["global_gain"]
        for checkpoint in checkpoints
    ]
    x2 = np.arange(2)
    ax2.bar(
        x2 - 0.17,
        mask_values,
        width=0.34,
        color=COLORS["cdf"],
        edgecolor="white",
        label="Thermal mask",
    )
    ax2.bar(
        x2 + 0.17,
        global_values,
        width=0.34,
        color=COLORS["neutral"],
        edgecolor="white",
        label="Global injection",
    )
    for checkpoint, mask_value, global_value in zip(
        checkpoints, mask_values, global_values
    ):
        source_rows.append(
            {
                "panel": "b",
                "checkpoint": checkpoint,
                "thermal_mask_gain_ap50_points": round(mask_value, 6),
                "global_gain_ap50_points": round(global_value, 6),
                "mask_minus_global_points": round(mask_value - global_value, 6),
            }
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["Best", "Last"])
    ax2.set_ylabel("CDF AP50 gain (points)")
    ax2.set_title("Thermal-mask ablation", loc="left", pad=7)
    ax2.set_ylim(0, 0.8)
    ax2.legend(loc="upper left")
    add_panel_label(ax2, "b", x=-0.23, y=1.02)

    ax3 = fig.add_subplot(grid[1, :])
    datasets = [
        ("llvip", "LLVIP"),
        ("m3fd", "M3FD"),
        ("dronevehicle_test", "DroneVehicle"),
    ]
    y3 = np.arange(len(datasets))
    associations = [
        ("thermal_rho", "Thermal top-hat", COLORS["spg"], "o"),
        ("visible_rho", "Visible luminance", COLORS["visible"], "s"),
        ("thermal_mask_rho", "CDF-mask activity", COLORS["cdf"], "^"),
    ]
    for dataset_index, (dataset, display) in enumerate(datasets):
        row = thermal_data[dataset]["correlation_with_switch_fraction"]
        for association, label, color, marker in associations:
            value = row[association]
            ax3.scatter(
                value,
                dataset_index,
                color=color,
                marker=marker,
                s=42,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )
            source_rows.append(
                {
                    "panel": "c",
                    "dataset": display,
                    "association": label,
                    "spearman_rho": value,
                    "p_value": row[association.replace("_rho", "_p")],
                }
            )
    ax3.axvline(0, color=COLORS["dark"], linewidth=0.8)
    ax3.set_yticks(y3)
    ax3.set_yticklabels([display for _, display in datasets])
    ax3.invert_yaxis()
    ax3.set_xlim(-0.55, 0.4)
    ax3.set_xlabel("Spearman rho with switch fraction")
    ax3.set_title("Thermal association is not a universal switching rule", loc="left", pad=7)
    ax3.legend(
        handles=[
            Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markersize=5, label=label)
            for _, label, color, marker in associations
        ],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
    )
    add_panel_label(ax3, "c", x=-0.09, y=1.04)

    write_csv(
        "figure4_source.csv",
        [
            "panel",
            "model",
            "class",
            "gain_ap50_points",
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
    save_figure(fig, "fig4_thermal_class_boundary", dpi=PUBLICATION_DPI)


if __name__ == "__main__":
    main()
