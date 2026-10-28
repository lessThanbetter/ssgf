"""Figure 10: direct comparison with the Always CDF baseline."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ipt_figure_common import (
    COLORS,
    add_panel_label,
    apply_style,
    clean_axis,
    load_json,
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


DATASET_LABELS = {
    "llvip": "LLVIP",
    "m3fd": "M3FD",
    "dronevehicle_test": "DroneVehicle",
}
DATASET_COLORS = {
    "llvip": COLORS["llvip"],
    "m3fd": COLORS["m3fd"],
    "dronevehicle_test": COLORS["drone"],
}
CHECKPOINT_LABELS = {
    "yolo11": "LLVIP · YOLO11n",
    "agnostic": "LLVIP · Modality-agnostic",
    "yolo26_s0": "LLVIP · YOLO26n s0",
    "rtdetr": "LLVIP · RT-DETR-l",
    "yolo26_s1": "LLVIP · YOLO26n s1",
    "yolo26_s2": "LLVIP · YOLO26n s2",
    "yolo8_s0": "LLVIP · YOLOv8n s0",
    "yolo8_s1": "LLVIP · YOLOv8n s1",
    "yolo26_m3fd6": "M3FD · YOLO26n",
    "yolo8n_m3fd6": "M3FD · YOLOv8n",
    "best": "DV · OBB best",
    "last": "DV · OBB last",
}
ORDER = list(CHECKPOINT_LABELS)


def main() -> None:
    apply_style()
    data = load_json("csecf_vs_cdf_paired.json")
    source_rows = []

    fig = plt.figure(figsize=(6.35, 5.45))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.15, 0.85],
        height_ratios=[0.78, 1.55],
        hspace=0.36,
        wspace=0.38,
        left=0.20,
        right=0.96,
        top=0.92,
        bottom=0.10,
    )

    ax = fig.add_subplot(grid[0, 0])
    scopes = [
        ("LLVIP", data["per_dataset"]["llvip"]["point_csecf_minus_cdf"], data["per_dataset"]["llvip"]["fixed_models_group_ci95"]),
        ("M3FD", data["per_dataset"]["m3fd"]["point_csecf_minus_cdf"], data["per_dataset"]["m3fd"]["fixed_models_group_ci95"]),
        ("DroneVehicle", data["per_dataset"]["dronevehicle_test"]["point_csecf_minus_cdf"], data["per_dataset"]["dronevehicle_test"]["fixed_models_group_ci95"]),
        ("Balanced", data["dataset_balanced"]["point_csecf_minus_cdf"], data["dataset_balanced"]["run_cluster_by_group_ci95"]),
    ]
    for index, (label, point, interval) in enumerate(scopes):
        estimate = 100 * point
        low, high = [100 * value for value in interval]
        color = COLORS["gain"] if estimate >= 0 else COLORS["loss"]
        ax.plot([low, high], [index, index], color=color, linewidth=1.2)
        ax.scatter(estimate, index, s=44 if label == "Balanced" else 28, color=color, edgecolor="white", linewidth=0.5, zorder=4)
        ax.text(high + 0.55, index, f"{estimate:+.1f}", ha="left", va="center", fontsize=5.4, color=color)
        source_rows.append(
            {
                "panel": "a",
                "scope": label,
                "delta_ap50_points": round(estimate, 6),
                "ci_low_ap50_points": round(low, 6),
                "ci_high_ap50_points": round(high, 6),
            }
        )
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_yticks(range(len(scopes)))
    ax.set_yticklabels([label for label, _, _ in scopes], fontsize=5.9)
    ax.invert_yaxis()
    ax.set_xlim(-2.2, 8.2)
    ax.set_xticks([-2, 0, 2, 4, 6, 8])
    ax.set_xlabel("CS-ECF minus Always CDF (points)", fontsize=5.9)
    ax.set_title("Dataset-level contrast", loc="left", pad=6, fontsize=6.8)
    clean_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0, pad=3)
    add_panel_label(ax, "a", x=-0.27, y=1.04)

    ax2 = fig.add_subplot(grid[0, 1])
    for dataset, color in DATASET_COLORS.items():
        names = [
            name
            for name in ORDER
            if data["per_checkpoint"][name]["dataset"] == dataset
        ]
        entries = [data["per_checkpoint"][name] for name in names]
        x = np.asarray([100 * row["always_cdf_ap50"] for row in entries])
        y = np.asarray([100 * row["csecf_ap50"] for row in entries])
        ax2.scatter(x, y, s=27, color=color, edgecolor="white", linewidth=0.5, label=DATASET_LABELS[dataset], zorder=3)
        for name, x_value, y_value in zip(names, x, y):
            source_rows.append(
                {
                    "panel": "b",
                    "scope": name,
                    "always_cdf_ap50_points": round(x_value, 6),
                    "csecf_ap50_points": round(y_value, 6),
                }
            )
    lower, upper = 60, 92
    ax2.plot([lower, upper], [lower, upper], color=COLORS["line"], linewidth=0.75, linestyle=(0, (2, 1.4)))
    ax2.set_xlim(lower, upper)
    ax2.set_ylim(lower, upper)
    ax2.set_xticks([60, 70, 80, 90])
    ax2.set_yticks([60, 70, 80, 90])
    ax2.set_xlabel("Always CDF AP50", fontsize=5.9)
    ax2.set_ylabel("CS-ECF AP50", fontsize=5.9)
    ax2.set_title("Checkpoint pairing", loc="left", pad=6, fontsize=6.8)
    clean_axis(ax2, grid_axis="both")
    ax2.legend(
        loc="upper left",
        fontsize=5.0,
        handletextpad=0.3,
        labelspacing=0.35,
    )
    add_panel_label(ax2, "b", x=-0.30, y=1.04)

    ax3 = fig.add_subplot(grid[1, :])
    for index, name in enumerate(ORDER):
        row = data["per_checkpoint"][name]
        point = 100 * row["csecf_minus_cdf"]
        low, high = [100 * value for value in row["csecf_minus_cdf_group_ci95"]]
        color = COLORS["gain"] if point >= 0 else COLORS["loss"]
        ax3.plot([low, high], [index, index], color=color, linewidth=1.0, alpha=0.85)
        ax3.scatter(point, index, s=28, color=color, edgecolor="white", linewidth=0.45, zorder=4)
        ax3.text(high + 0.28, index, f"{point:+.1f}", ha="left", va="center", fontsize=5.1, color=color)
        source_rows.append(
            {
                "panel": "c",
                "scope": name,
                "delta_ap50_points": round(point, 6),
                "ci_low_ap50_points": round(low, 6),
                "ci_high_ap50_points": round(high, 6),
            }
        )
    for boundary in (7.5, 9.5):
        ax3.axhline(boundary, color="#DCE2E8", linewidth=0.55)
    ax3.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax3.set_yticks(range(len(ORDER)))
    ax3.set_yticklabels([CHECKPOINT_LABELS[name] for name in ORDER], fontsize=5.4)
    ax3.invert_yaxis()
    ax3.set_xlim(-3.8, 5.2)
    ax3.set_xticks([-3, -2, -1, 0, 1, 2, 3, 4])
    ax3.set_xlabel("CS-ECF minus Always CDF AP50 (points)", fontsize=5.9)
    ax3.set_title("Paired checkpoint contrasts", loc="left", pad=6, fontsize=6.8)
    clean_axis(ax3, grid_axis="x")
    ax3.tick_params(axis="y", length=0, pad=3)
    add_panel_label(ax3, "c", x=-0.24, y=1.03)

    write_csv(
        "figure10_source.csv",
        [
            "panel",
            "scope",
            "delta_ap50_points",
            "ci_low_ap50_points",
            "ci_high_ap50_points",
            "always_cdf_ap50_points",
            "csecf_ap50_points",
        ],
        source_rows,
    )
    save_figure(fig, "fig10_cdf_comparison", dpi=600)


if __name__ == "__main__":
    main()
